import os
import sys
import io
import uuid
import base64
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import aiofiles
from pydantic import BaseModel
import onnxruntime as ort
import yaml
import json
from datetime import datetime

# Import configuration and CLO value processing functions
from config import config
from clo_processor import map_detections_to_clo

# Setup logging for detector service
import logging
import os

def setup_detector_logging():
    """Setup logging for the detector service with relative paths."""
    # Get detector directory for relative path resolution
    detector_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configure logging
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if config.logging.enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler - only create if file logging is explicitly enabled
    if config.logging.enable_file and config.logging.file and config.logging.file is not None:
        # Resolve log file path relative to detector directory
        log_file_path = os.path.join(detector_dir, config.logging.file)
        
        # Only create log directory if file logging is enabled
        log_dir = os.path.dirname(log_file_path)
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Setup logging
detector_logger = setup_detector_logging()



# Initialize FastAPI app
app = FastAPI(title="RT-DETR v2 ONNX Inference API", description="API for object detection using RT-DETR v2 ONNX")

# Model configuration
class ModelConfig(BaseModel):
    onnx_file: str = config.model.onnx_file
    img_size: int = config.model.img_size
    conf_thres: float = config.model.conf_thres
    device: str = config.model.device
    categories_file: str = config.paths.categories_file

# Detection result
class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    clo_value: Optional[float] = None

# Response model
class InferenceResponse(BaseModel):
    detections: List[DetectionResult]
    total_clo_value: Optional[float] = None

# Global model variables
model_session = None
class_names = []

def save_detection_result(result: InferenceResponse, image_source: str = "unknown") -> Optional[str]:
    """Save detection result to JSON file in a date-based folder
    
    Args:
        result: Detection result to save
        image_source: Source image name or identifier
        
    Returns:
        Path to saved file or None if saving failed
    """
    # Check if saving is enabled in config
    if not config.save_results.enabled:
        detector_logger.warning("Saving detection results is disabled in config")
        return None
    
    try:
        # Get detector directory for relative path resolution
        detector_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Generate filename from image source
        base_name = os.path.splitext(os.path.basename(image_source))[0]
        filename = f"{base_name}.json"
        
        # Get current date for folder name
        current_date = datetime.now().strftime("%Y%m%d")
        
        # Resolve output directory path based on config
        config_output_dir = config.save_results.output_dir
        
        # Handle different path formats
        if config_output_dir.startswith('./'):
            # Relative to detector directory
            base_output_dir = os.path.join(detector_dir, config_output_dir[2:])
        elif '/' not in config_output_dir and '\\' not in config_output_dir:
            # Just a directory name, make it relative to detector dir
            base_output_dir = os.path.join(detector_dir, config_output_dir)
        elif config_output_dir.startswith('detector/'):
            # Remove 'detector/' prefix and make relative to detector dir
            base_output_dir = os.path.join(detector_dir, config_output_dir[9:])
        else:
            # Use as is (absolute path or other relative path)
            base_output_dir = config_output_dir
        
        # Create date-based directory inside output directory
        output_dir = os.path.join(base_output_dir, current_date)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create full file path
        file_path = os.path.join(output_dir, filename)
        
        # Prepare data for saving
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "image_source": image_source,
            "model_config": {
                "img_size": config.model.img_size,
                "conf_thres": config.model.conf_thres,
                "device": config.model.device
            },
            "detections": [detection.model_dump() for detection in result.detections],
            "total_clo_value": result.total_clo_value,
            "total_detections": len(result.detections)
        }
        
        # Save to JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        detector_logger.info(f"Detection: {os.path.basename(file_path)} ({len(result.detections)} objects)")
        return file_path
        
    except Exception as e:
        detector_logger.error(f"Error saving detection result: {e}")
        return None

# Load class names from YAML file
def load_class_names(yaml_file):
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract category names and create a list indexed by ID
        categories = data.get('categories', {})
        
        # Find the maximum ID to determine list size
        max_id = max(categories.keys()) if categories else 0
        
        # Create a list where index corresponds to class ID
        labels = [''] * (max_id + 1)  # Initialize with empty strings
        
        for class_id, class_name in categories.items():
            labels[class_id] = class_name
            
        return labels
    except Exception as e:
        detector_logger.error(f"Error loading labels from {yaml_file}: {e}")
        return []

# Load ONNX model
def load_model(config: ModelConfig):
    global model_session, class_names
    
    if model_session is None:
        try:
            # Check if the model file exists
            model_path = config.onnx_file
            
            # Try different paths for the model file
            if not os.path.exists(model_path):
                # Try relative to detector directory
                detector_dir = os.path.dirname(os.path.abspath(__file__))
                alt_model_path = os.path.join(detector_dir, "model.onnx")
                if os.path.exists(alt_model_path):
                    model_path = alt_model_path
                    detector_logger.info(f"Using model from detector directory: {model_path}")
                else:
                    raise HTTPException(status_code=404, detail=f"Model file not found: {model_path} or {alt_model_path}")
            
            # Set execution providers
            if config.device.lower() == 'gpu':
                try:
                    providers = [
                        ('CUDAExecutionProvider', {}),
                        ('CPUExecutionProvider', {})
                    ]
                    detector_logger.info("Attempting GPU acceleration with CUDA")
                except:
                    providers = [('CPUExecutionProvider', {})]
                    detector_logger.info("GPU not available, falling back to CPU")
            else:
                providers = [('CPUExecutionProvider', {})]
                detector_logger.info("Using CPU execution (stable and reliable)")
            
            # Create ONNX Runtime session
            model_session = ort.InferenceSession(model_path, providers=providers)
            
            # Log available providers and current device
            detector_logger.info(f"Available providers: {ort.get_available_providers()}")
            detector_logger.info(f"Current provider: {model_session.get_providers()}")
            
            # Load class names
            if os.path.exists(config.categories_file):
                class_names = load_class_names(config.categories_file)
                detector_logger.info(f"Loaded {len(class_names)} class names from {config.categories_file}")
            else:
                # Try relative to detector directory
                detector_dir = os.path.dirname(os.path.abspath(__file__))
                alt_categories_file = os.path.join(detector_dir, "categories.yaml")
                if os.path.exists(alt_categories_file):
                    class_names = load_class_names(alt_categories_file)
                    detector_logger.info(f"Loaded {len(class_names)} class names from detector directory: {alt_categories_file}")
                else:
                    detector_logger.warning(f"Categories file not found: {config.categories_file} or {alt_categories_file}")
                    class_names = []
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Preprocess image
def preprocess_image(image_data, target_size=(640, 640)):
    """Preprocess image data"""
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    orig_w, orig_h = img.size
    
    # Resize image
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Transpose from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
    img_chw = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension: (3, 640, 640) -> (1, 3, 640, 640)
    img_tensor = np.expand_dims(img_chw, axis=0)
    
    return img_tensor, (orig_w, orig_h)

# Process image for inference
async def process_image(image_data: bytes, model_config: ModelConfig) -> InferenceResponse:
    try:
        # Load model if not already loaded
        load_model(model_config)
        
        # Preprocess image
        im_data, (orig_w, orig_h) = preprocess_image(image_data, (model_config.img_size, model_config.img_size))
        orig_size = np.array([[orig_w, orig_h]], dtype=np.int64)
        
        # Run inference
        output = model_session.run(
            output_names=None,
            input_feed={'images': im_data, "orig_target_sizes": orig_size}
        )
        
        labels, boxes, scores = output
        
        # Process detections
        detections = []
        for i in range(len(labels[0])):
            label = int(labels[0][i])
            score = float(scores[0][i])
            box = boxes[0][i]
            
            # Apply confidence threshold
            if score >= model_config.conf_thres:
                # Get class name
                class_name = class_names[label] if label < len(class_names) and class_names[label] else f"Class_{label}"
                
                detections.append(DetectionResult(
                    class_id=label,
                    class_name=class_name,
                    confidence=score,
                    bbox=[float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                ))
        
        # Apply CLO value mapping post-processing
        if config.clo.enabled:
            detections, total_clo_value = map_detections_to_clo(detections, config.clo.values_file)
        else:
            total_clo_value = None
        
        return InferenceResponse(
            detections=detections,
            total_clo_value=total_clo_value
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# API endpoints
@app.get("/")
async def root():
    return {"message": "RT-DETR v2 ONNX Inference API is running. Use /docs for API documentation."}

@app.get("/saved-detections")
async def list_saved_detections():
    """List saved detection result files"""
    if not config.save_results.enabled:
        raise HTTPException(status_code=400, detail="Detection result saving is disabled")
    
    try:
        detection_dir = config.save_results.output_dir
        if not os.path.exists(detection_dir):
            return {"saved_detections": [], "total": 0}
        
        files = []
        for filename in os.listdir(detection_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(detection_dir, filename)
                file_stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size_bytes": file_stat.st_size,
                    "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created"], reverse=True)
        
        return {
            "saved_detections": files,
            "total": len(files),
            "output_directory": detection_dir
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing saved detections: {str(e)}")

@app.get("/detect-default", response_model=InferenceResponse)
async def detect_default(onnx_file: str = "model.onnx",
                      img_size: int = 640,
                      conf_thres: float = 0.6,
                      device: str = "cpu",
                      categories_file: str = "categories.yaml"):
    
    # Use default image if available
    default_image_path = config.paths.default_image
    if os.path.exists(default_image_path):
        with open(default_image_path, 'rb') as f:
            contents = f.read()
    else:
        raise HTTPException(status_code=404, detail=f"Default image not found: {default_image_path}")
    
    # Create model config
    model_config = ModelConfig(
        onnx_file=onnx_file,
        img_size=img_size,
        conf_thres=conf_thres,
        device=device,
        categories_file=categories_file
    )
    
    # Process image
    result = await process_image(contents, model_config)
    
    # Save detection result if enabled
    save_detection_result(result, "default_image")
    
    return result

@app.post("/detect", response_model=InferenceResponse)
async def detect(file: Optional[UploadFile] = None, 
                onnx_file: str = Form("model.onnx"),
                img_size: int = Form(640),
                conf_thres: float = Form(0.6),
                device: str = Form("cpu"),
                categories_file: str = Form("categories.yaml"),
                frame_name: Optional[str] = Form(None)):
    
    detector_logger.info(f"Received detection request with file: {file.filename if file else 'None'}, frame_name: {frame_name}")
    
    # Read image file or use default image
    if file is not None:
        try:
            contents = await file.read()
            detector_logger.debug(f"Successfully read file contents, size: {len(contents)} bytes")
        except Exception as e:
            detector_logger.error(f"Error reading uploaded file: {e}")
            raise HTTPException(status_code=400, detail=f"Error reading uploaded file: {str(e)}")
    else:
        # Use default image
        default_image_path = config.paths.default_image
        detector_logger.debug(f"No file provided, using default image: {default_image_path}")
        if os.path.exists(default_image_path):
            with open(default_image_path, 'rb') as f:
                contents = f.read()
            detector_logger.debug(f"Read default image, size: {len(contents)} bytes")
        else:
            detector_logger.error(f"Default image not found: {default_image_path}")
            raise HTTPException(status_code=404, detail=f"Default image not found: {default_image_path}")
    
    # Create model config
    model_config = ModelConfig(
        onnx_file=onnx_file,
        img_size=img_size,
        conf_thres=conf_thres,
        device=device,
        categories_file=categories_file
    )
    detector_logger.debug(f"Created model config: {model_config}")
    
    # Process image
    detector_logger.info("Processing image for detection...")
    result = await process_image(contents, model_config)
    detector_logger.info(f"Image processed, found {len(result.detections)} detections")
    
    # Determine image source for saving - prioritize frame_name if provided
    if frame_name:
        image_source = frame_name
        detector_logger.debug(f"Using provided frame_name as image source: {image_source}")
    elif file and file.filename:
        image_source = file.filename
        detector_logger.debug(f"Using file.filename as image source: {image_source}")
    else:
        image_source = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        detector_logger.debug(f"Generated timestamp as image source: {image_source}")
    
    # Save detection result if enabled
    saved_path = save_detection_result(result, image_source)
    if saved_path:
        detector_logger.info(f"Detection result saved to: {saved_path}")
    else:
        detector_logger.warning("Failed to save detection result")
    
    return result

if __name__ == "__main__":
    uvicorn.run("detector:app", host=config.api.host, port=config.api.port, reload=config.api.reload)
