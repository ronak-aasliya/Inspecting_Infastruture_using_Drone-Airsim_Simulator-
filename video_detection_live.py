# scripts/video_detection_live.py - Live Video Detection with Real-time Display

import sys
import cv2
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crack_detector import TrainedCrackDetector

def video_detection_live(video_path, confidence=0.4):

    # Check if video exists
    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        return
    
    logger.info(f"\nVideo: {video_path}")
    
    # Load detector
    try:
        detector = TrainedCrackDetector(
            model_path='best.pt',
            device='cuda'
        )
        logger.info("Detector loaded")
    except Exception as e:
        logger.error(f"Detector failed: {e}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error("Cannot open video!")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"  FPS: {fps:.2f}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Duration: {total_frames/fps:.2f} seconds")
    

    logger.info("\nControls:")
    logger.info("  SPACE - Pause/Resume")
    logger.info("  Q - Quit")
    logger.info("  S - Save frame")
    logger.info("  +/- - Adjust confidence")
    logger.info("  F - Toggle fullscreen")
    
    frame_count = 0
    total_anomalies = 0
    anomalies_per_frame = []
    paused = False
    fullscreen = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    logger.info("\nVideo finished!")
                    break
                
                frame_count += 1
                
                # Run detection
                try:
                    detection_results = detector.detect(frame, conf=confidence)
                    anomaly_count = detection_results['anomaly_count']
                    frame_annotated = detection_results['image_annotated']
                    cracks = detection_results['cracks']
                    total_anomalies += anomaly_count
                    anomalies_per_frame.append(anomaly_count)
                    
                    if anomaly_count > 0:
                        logger.info(f"Frame {frame_count}: {anomaly_count} crack(s) detected")
                    
                except Exception as e:
                    logger.error(f"Detection error on frame {frame_count}: {e}")
                    frame_annotated = frame.copy()
                    anomaly_count = 0
                
                # Add text overlays
                overlay = frame_annotated.copy()
                
                # Frame info
                frame_text = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(overlay, frame_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Confidence
                conf_text = f"Confidence: {confidence:.2f}"
                cv2.putText(overlay, conf_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Detection status
                if anomaly_count > 0:
                    status_text = f"CRACKS FOUND! ({anomaly_count})"
                    color = (0, 0, 255)  # Red
                else:
                    status_text = "No cracks detected"
                    color = (0, 255, 0)  # Green
                
                cv2.putText(overlay, status_text, (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
                
                # Playback status
                status_pause = "PAUSED" if paused else "PLAYING"
                cv2.putText(overlay, status_pause, (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                # Total stats
                stats_text = f"Total cracks: {total_anomalies}"
                cv2.putText(overlay, stats_text, (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Time info
                current_time = frame_count / fps
                total_time = total_frames / fps
                time_text = f"Time: {current_time:.1f}s / {total_time:.1f}s"
                cv2.putText(overlay, time_text, (10, 230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Instructions
                instructions = [
                ]
                for i, instr in enumerate(instructions):
                    y = overlay.shape[0] - 20
                    cv2.putText(overlay, instr, (10, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Resize for display
                h, w = overlay.shape[:2]
                if w > 1400 or h > 900:
                    scale = min(1400 / w, 900 / h)
                    overlay = cv2.resize(overlay, (int(w*scale), int(h*scale)))
                
                # Display
                window_name = "Live Video Detection"
                cv2.imshow(window_name, overlay)
                
                # Handle fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_NORMAL)
                
                # Wait for frame time (play at correct FPS)
                frame_delay = int(1000 / fps)  # milliseconds
            else:
                # When paused, just wait for input
                frame_delay = 100
                cv2.imshow("Live Video Detection", overlay)
            
            # Check for key press
            key = cv2.waitKey(frame_delay) & 0xFF
            
            if key == ord('q'):  # Quit
                logger.info("\nQuitting...")
                break
            
            elif key == ord(' '):  # Pause/Resume
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                logger.info(f"{status} - Press SPACE to toggle")
            
            elif key == ord('s'):  # Save frame
                filename = f"detection_live_{frame_count:04d}.jpg"
                cv2.imwrite(filename, overlay)
                logger.info(f"Saved: {filename}")
            
            elif key == ord('+') or key == ord('='):  # Increase confidence
                confidence = min(1.0, confidence + 0.05)
                logger.info(f"Confidence: {confidence:.2f}")
            
            elif key == ord('-') or key == ord('_'):  # Decrease confidence
                confidence = max(0.0, confidence - 0.05)
                logger.info(f"Confidence: {confidence:.2f}")
            
            elif key == ord('f') or key == ord('F'):  # Toggle fullscreen
                fullscreen = not fullscreen
                logger.info(f"Fullscreen: {'ON' if fullscreen else 'OFF'}")
    
    except KeyboardInterrupt:
        logger.info("\nInterrution")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"  • Frames processed: {frame_count}")
        logger.info(f"  • Total cracks detected: {total_anomalies}")
        
        if frame_count > 0:
            avg_cracks = total_anomalies / frame_count
            logger.info(f"Average: {avg_cracks:.2f} cracks/frame")
            
            if anomalies_per_frame:
                max_cracks = max(anomalies_per_frame)
                max_frame = anomalies_per_frame.index(max_cracks) + 1
                logger.info(f"Max cracks: {max_cracks} (frame {max_frame})")

def main():
    
    # Check command line args
    if len(sys.argv) < 2:
        logger.error("No video file provided!")
        print("\nUsage: python scripts/video_detection_live.py <video_path> [confidence]")
        return
    
    video_path = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    
    # Validate confidence
    if not 0.0 <= confidence <= 1.0:
        logger.error(f"Invalid confidence: {confidence} (must be 0.0-1.0)")
        return
    
    logger.info(f"Confidence threshold: {confidence}")
    
    # Run detection
    video_detection_live(video_path, confidence=confidence)

if __name__ == "__main__":
    main()
