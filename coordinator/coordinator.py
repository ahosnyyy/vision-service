#!/usr/bin/env python3
"""
Vision Service Coordinator

Orchestrates the recorder and detector services, manages frame processing,
and handles the overall lifecycle of the vision service.
"""

import threading
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from shared.buffer import FrameBuffer
from shared.utils import setup_logging, load_config
from recorder.recorder import Recorder
from detector.detector_client import Detector


class VisionCoordinator:
    """Main coordinator for the vision service."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the coordinator with configuration."""
        self.config = load_config(config_path)
        self.logger = setup_logging("coordinator")
        
        # Service components
        self.recorder: Optional[Recorder] = None
        self.detector: Optional[Detector] = None
        
        # Shared resources
        self.frame_buffer: Optional[FrameBuffer] = None
        
        # Control flags
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Configuration
        self.detection_interval = self.config.get("detection_interval", 10)  # Process every Nth frame
        self.buffer_size = self.config.get("buffer_size", 100)
        self.processing_delay = self.config.get("processing_delay", 0.1)  # seconds
        
        self.logger.info("Vision Coordinator initialized")
    
    def start(self) -> bool:
        """Start the vision service."""
        try:
            self.logger.info("Starting Vision Service...")
            
            # Initialize shared buffer
            self.frame_buffer = FrameBuffer(maxlen=self.buffer_size)
            self.logger.info(f"Frame buffer initialized with size {self.buffer_size}")
            
            # Start recorder service
            self.recorder = Recorder(
                config_path=self.config.get("recorder_config", "recorder/config.yaml"),
                frame_buffer=self.frame_buffer
            )
            self.recorder.start()
            self.logger.info("Recorder service started")
            
            # Start detector service
            self.detector = Detector(
                config_path=self.config.get("detector_config", "detector/config.yaml")
            )
            self.detector.start()
            self.logger.info("Detector service started")
            
            # Start frame processing
            self.running = True
            self.processing_thread = threading.Thread(
                target=self._process_frames,
                daemon=True,
                name="FrameProcessor"
            )
            self.processing_thread.start()
            self.logger.info("Frame processing started")
            
            self.logger.info("Vision Service started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Vision Service: {e}")
            self.stop()
            return False
    
    def stop(self) -> None:
        """Stop the vision service."""
        self.logger.info("Stopping Vision Service...")
        self.running = False
        
        # Stop frame processing
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            self.logger.info("Frame processing stopped")
        
        # Stop detector service
        if self.detector:
            self.detector.stop()
            self.logger.info("Detector service stopped")
        
        # Stop recorder service
        if self.recorder:
            self.recorder.stop()
            self.logger.info("Recorder service stopped")
        
        # Clear buffer
        if self.frame_buffer:
            self.frame_buffer.clear()
        
        self.logger.info("Vision Service stopped")
    
    def _process_frames(self) -> None:
        """Main frame processing loop."""
        frame_count = 0
        
        while self.running:
            try:
                if not self.frame_buffer or self.frame_buffer.empty():
                    time.sleep(self.processing_delay)
                    continue
                
                # Get frame from buffer
                frame_data = self.frame_buffer.get()
                if frame_data is None:
                    continue
                
                frame_count += 1
                
                # Decide whether to process this frame
                if frame_count % self.detection_interval == 0:
                    self.logger.debug(f"Processing frame {frame_count}")
                    
                    # Send frame to detector via HTTP API
                    if self.detector and self.detector.is_ready():
                        try:
                            detection_result = self.detector.detect_frame(frame_data)
                            if detection_result:
                                self.logger.info(f"Detection completed for frame {frame_count}")
                                # Handle detection results here
                        except Exception as e:
                            self.logger.error(f"Detection failed for frame {frame_count}: {e}")
                    else:
                        self.logger.warning("Detector not ready, skipping frame")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(self.processing_delay)
                
            except Exception as e:
                self.logger.error(f"Error in frame processing loop: {e}")
                time.sleep(self.processing_delay)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the vision service."""
        status = {
            "running": self.running,
            "recorder_active": self.recorder.is_running() if self.recorder else False,
            "detector_active": self.detector.is_running() if self.detector else False,
            "buffer_size": len(self.frame_buffer) if self.frame_buffer else 0,
            "buffer_capacity": self.buffer_size,
            "detection_interval": self.detection_interval
        }
        return status
    
    def is_running(self) -> bool:
        """Check if the service is running."""
        return self.running


def main():
    """Main entry point for the vision service."""
    coordinator = VisionCoordinator()
    
    try:
        if coordinator.start():
            # Keep the service running
            while coordinator.is_running():
                time.sleep(1)
        else:
            print("Failed to start Vision Service")
            return 1
            
    except KeyboardInterrupt:
        print("\nShutting down Vision Service...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    finally:
        coordinator.stop()
    
    return 0


if __name__ == "__main__":
    exit(main())
