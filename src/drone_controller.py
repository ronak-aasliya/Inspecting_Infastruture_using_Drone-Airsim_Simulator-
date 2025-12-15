# src/drone_controller.py - FINAL FIX (Compressed Image Handling)

import airsim
import numpy as np
import cv2
import logging
import time
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class DroneController:
    
    def __init__(self):
        self.client = None
        self.is_connected = False
        self.home_location = None
        
    def connect(self) -> bool:

        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.is_connected = True
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        if self.client:
            self.client = None
            self.is_connected = False
            logger.info("Disconnected")
    
    def enable_api_control(self) -> bool:
        try:
            self.client.enableApiControl(True)
            return True
        except Exception as e:
            logger.error(f"Failed API control: {e}")
            return False
    
    def arm(self) -> bool:
        try:
            self.client.armDisarm(True)
            logger.info("Drone armed")
            return True
        except Exception as e:
            logger.error(f"Failed to arm drone: {e}")
            return False
    
    def disarm(self) -> bool:

        try:
            self.client.armDisarm(False)
            logger.info("Drone disarmed")
            return True
        except Exception as e:
            logger.error(f"Failed to disarm drone: {e}")
            return False
    
    def takeoff(self, altitude: float = 10.0) -> bool:

        try:
            logger.info(f"Taking off to {altitude}m")
            self.client.takeoffAsync().join()
            
            # Get home position
            state = self.client.getMultirotorState()
            self.home_location = state.kinematics_estimated.position
            
            logger.info("Takeoff complete")
            return True
        except Exception as e:
            logger.error(f"Takeoff failed: {e}")
            return False
    
    def land(self) -> bool:
        try:
            logger.info("Landing drone")
            self.client.landAsync().join()
            self.disarm()
            return True
        except Exception as e:
            logger.error(f"Landing failed: {e}")
            return False
    
    def fly_to_position(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float = 5.0
    ) -> bool:
        try:
            self.client.moveToPositionAsync(x, y, z, velocity).join()
            return True
        except Exception as e:
            logger.error(f"Failed to fly to position: {e}")
            return False
    
    def fly_waypoint_mission(
        self,
        waypoints: list,
        velocity: float = 5.0,
        lookahead: float = 5.0,
        adaptive_lookahead: bool = True
    ) -> bool:

        try:
            logger.info(f"Starting waypoint mission with {len(waypoints)} waypoints")
            
            waypoint_list = [
                airsim.Vector3r(x, y, z) for x, y, z in waypoints
            ]
            
            self.client.moveOnPathAsync(
                waypoint_list,
                velocity,
                timeout_sec=float('inf'),
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(False, 0),
                lookahead=lookahead,
                adaptive_lookahead=adaptive_lookahead
            ).join()
            
            logger.info("Waypoint mission completed")
            return True
        except Exception as e:
            logger.error(f"Waypoint mission failed: {e}")
            return False
    
    def return_to_launch(self) -> bool:

        if self.home_location is None:
            logger.warning("Home location not set")
            return False
        
        try:
            logger.info("Returning to launch")
            self.fly_to_position(
                x=self.home_location.x_val,
                y=self.home_location.y_val,
                z=self.home_location.z_val,
                velocity=5.0
            )
            return True
        except Exception as e:
            logger.error(f"Return to launch failed: {e}")
            return False
    
    def get_drone_state(self) -> Dict:
        try:
            state = self.client.getMultirotorState()
            kin = state.kinematics_estimated
            
            return {
                'position': (kin.position.x_val, kin.position.y_val, kin.position.z_val),
                'velocity': (kin.linear_velocity.x_val, kin.linear_velocity.y_val, kin.linear_velocity.z_val),
                'landed': state.landed_state == airsim.LandedState.Landed if hasattr(state, 'landed_state') else False,
            }
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return None
    
    def get_position(self) -> Tuple[float, float, float]:
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            return (pos.x_val, pos.y_val, pos.z_val)
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return None
    
    def capture_image(self, camera_name: int = 0) -> np.ndarray:
        
        try:
            # Get image from camera
            response = self.client.simGetImage(camera_name, airsim.ImageType.Scene)
            
            if response is None or len(response) == 0:
                logger.warning("Empty image response")
                return None
            
            img_array = np.frombuffer(response, dtype=np.uint8)
            

            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                logger.warning("Failed to decode image")
                return None
            
            logger.debug(f"Captured image: {img_bgr.shape}")
            return img_bgr
            
        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def hover(self, duration: float = 5.0) -> bool:

        try:
            logger.info(f"Hovering for {duration}s")
            self.client.hoverAsync().join()
            time.sleep(duration)
            return True
        except Exception as e:
            logger.error(f"Hover failed: {e}")
            return False
