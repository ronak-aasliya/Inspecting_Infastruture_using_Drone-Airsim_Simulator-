import sys
import cv2
from pathlib import Path
import logging
import time
import airsim

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drone_controller import DroneController
from src.crack_detector import TrainedCrackDetector

def live_detection():
    
    drone = DroneController()
    
    if not drone.connect():
        logger.error("Connection failed!")
        return
    logger.info("Connected")
    
    if not drone.enable_api_control():
        logger.error("API control failed!")
        drone.disconnect()
        return
    logger.info("API enabled")
    
    if not drone.arm():
        logger.error("Arm failed!")
        drone.disconnect()
        return
    logger.info("Armed")
    
    state = drone.get_drone_state()
    if state is None:
        logger.error("Can't read state!")
        drone.disconnect()
        return
    logger.info(f"State: {state}")


    

    try:
        detector = TrainedCrackDetector(model_path='best.pt', device=None)
        logger.info("Detector intialized")
    except Exception as e:
        logger.error(f"Detector failed: {e}")
        drone.disconnect()
        return
    
    
    
    frame_count = 0
    fullscreen = False
    confidence_th = 0.5
    total_crack = 0


    altitude = 10.0
    if not drone.takeoff(altitude=altitude):
        drone.disconnect()
        return

    # 2) Build square waypoints in NED
    z = -altitude  
    side = 20.0    

    waypoints = [
        (0.0,   0.0,   z),   # intial postion
        (side,  0.0,   z),
        (side,  side,  z),
        (0.0,   side,  z),
        (0.0,   0.0,   z),   
    ]

    speed = 5.0
    drone.client.moveOnPathAsync(
        [airsim.Vector3r(x, y, z) for (x, y, z) in waypoints],
        speed,
        timeout_sec=float('inf'),
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(False, 0),
        lookahead=5.0,
        adaptive_lookahead=1.0
    )
    
    try:
        while True:
            # Capturing image
            img = drone.capture_image()
            
            if img is None:
                continue
            
            frame_count += 1
            
            #detection
            try:
                detection_results = detector.detect(img, conf=confidence_th)
                anomaly_count = detection_results['anomaly_count']
                img_annotated = detection_results['image_annotated']
                total_crack += anomaly_count
                
                if anomaly_count > 0:
                    logger.info(f"Crack Detected {frame_count}: {anomaly_count} crack(s)")
                
            except Exception as e:
                logger.error(f"Detection error: {e}")
                img_annotated = img.copy()
                anomaly_count = 0
            
            #current position
            pos = drone.get_position()
            
            #text overlays
            overlay = img_annotated.copy()
            
            
            
            # Frame and detection stats
            frame_text = f"Frame: {frame_count} | Cracks in frame: {anomaly_count} | Total: {total_crack}"
            cv2.putText(
                overlay, frame_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2
            )
            
            
            # Anomaly status (highlight if detected)
            if anomaly_count > 0:
                status_text = f"CRACK DETECTED!"
                color = (0, 0, 255)  # Red
            else:
                status_text = "No anomalies"
                color = (0, 255, 0)  # Green
            
            cv2.putText(
                overlay, status_text, (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 3
            )
            
            # Resize for display
            overlay = cv2.resize(overlay, (1920, 1080))  # Full

            
            # Display
            window_name = "Live Crack Detection"
            cv2.imshow(window_name, overlay)
            
            # Handle fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_NORMAL)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                logger.info("quit - disconnecting...")
                break
            elif key == ord('s'):  # Save frame
                filename = f"detection_frame_{frame_count:04d}.jpg"
                cv2.imwrite(filename, overlay)
                logger.info(f"Annotated frame saved: {filename}")
            elif key == ord('f'):  # Toggle fullscreen
                fullscreen = not fullscreen
                logger.info(f"Fullscreen: {'ON' if fullscreen else 'OFF'}")
            elif key == ord('+') or key == ord('='):  # Increase confidence
                confidence_th = min(1.0, confidence_th + 0.05)
                logger.info(f"Confidence threshold: {confidence_th:.2f}")
            elif key == ord('-') or key == ord('_'):  # Decrease confidence
                confidence_th = max(0.0, confidence_th - 0.05)
                logger.info(f"Confidence threshold: {confidence_th:.2f}")
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        drone.land()
        drone.disconnect()
        

if __name__ == "__main__":
    live_detection()
