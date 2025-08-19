import cv2
import yaml
import os
import time
from datetime import datetime
from pathlib import Path
import logging

class Recorder:
    def __init__(self, config_path="config.yaml"):
        """Initialize the recorder with configuration from YAML file."""
        # Setup logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Now load config (logger is available)
        self.config = self._load_config(config_path)
        self.cap = None
        self.output_folder = None
        self.frame_count = 0
        self.start_time = None
        
        # Create output directory
        self._setup_output_directory()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.logger.info(f"Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            self.logger.error(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing config file: {e}. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return default configuration if config file is not available."""
        return {
            'recorder': {
                'camera_index': 0,
                'fps': 30,
                'resolution': {
                    'width': 1280,
                    'height': 720
                },
                'image_format': 'jpg',
                'output_dir': './frames',
                'buffer_size': 100,
                'keep_disk_copy': False
            }
        }
    
    def _setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        output_path = Path(self.config['recorder']['output_dir'])
        output_path.mkdir(exist_ok=True)
        self.output_folder = output_path
        self.logger.info(f"Output directory: {self.output_folder}")
    
    def start_camera(self):
        """Initialize and start the camera."""
        try:
            self.cap = cv2.VideoCapture(self.config['recorder']['camera_index'])
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FPS, self.config['recorder']['fps'])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['recorder']['resolution']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['recorder']['resolution']['height'])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['recorder']['buffer_size'])
            
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            
            self.logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop and release the camera."""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.logger.info("Camera stopped")
    
    def capture_frame(self):
        """Capture a single frame from the camera."""
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Camera is not available")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Failed to capture frame")
            return None
        
        return frame
    
    def save_frame(self, frame):
        """Save a frame to the output directory."""
        if frame is None:
            return False
        
        # Only save to disk if keep_disk_copy is True
        if not self.config['recorder']['keep_disk_copy']:
            self.frame_count += 1
            return True
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"frame_{timestamp}.{self.config['recorder']['image_format']}"
        
        filepath = self.output_folder / filename
        
        # Save frame
        success = cv2.imwrite(str(filepath), frame)
        
        if success:
            self.frame_count += 1
            self.logger.debug(f"Frame saved: {filename}")
            return True
        else:
            self.logger.error(f"Failed to save frame: {filename}")
            return False
    
    def start_recording(self):
        """Start the recording process."""
        if not self.start_camera():
            return False
        
        self.start_time = time.time()
        self.frame_count = 0
        self.logger.info("Recording started")
        
        try:
            while True:
                # Check if we should stop recording
                if self._should_stop_recording():
                    break
                
                # Capture and save frame
                frame = self.capture_frame()
                if frame is not None:
                    self.save_frame(frame)
                
                # Control frame rate
                time.sleep(1.0 / self.config['recorder']['fps'])
                
        except KeyboardInterrupt:
            self.logger.info("Recording stopped by user")
        except Exception as e:
            self.logger.error(f"Recording error: {e}")
        finally:
            self.stop_camera()
            self._print_recording_summary()
    
    def _should_stop_recording(self):
        """Check if recording should stop based on configuration."""
        # With the new config structure, recording continues indefinitely
        # until manually stopped by user (Ctrl+C)
        return False
    
    def _print_recording_summary(self):
        """Print a summary of the recording session."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.info(f"Recording completed:")
            self.logger.info(f"  Duration: {duration:.2f} seconds")
            self.logger.info(f"  Frames captured: {self.frame_count}")
            self.logger.info(f"  Average FPS: {self.frame_count / duration:.2f}")
            self.logger.info(f"  Output folder: {self.output_folder}")

def main():
    """Main function to run the recorder."""
    recorder = Recorder()
    
    print("Recorder initialized. Starting recording immediately...")
    print("Press Ctrl+C to stop recording and exit.")
    try:
        recorder.start_recording()
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
