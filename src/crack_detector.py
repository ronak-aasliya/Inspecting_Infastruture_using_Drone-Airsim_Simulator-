
import os
import torch
import cv2
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except:
    pass

from ultralytics import YOLO

class TrainedCrackDetector:
    def __init__(self, model_path='best.pt', device='cuda'):
        
        self.device = device
        self.model = None
        
        try:
            self._load_with_patch(model_path)
        except Exception as e:
            logger.warning(f"failed: {e}")
            try:
                self._load_with_context(model_path)
            except Exception as e2:
                logger.error(f"failed: {e2}")
                raise RuntimeError(f"Failed to load model: {e2}")
    
    def _load_with_patch(self, model_path):
        # Patch torch.load globally
        original_load = torch.load
        
        def patched_load(f, *args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(f, *args, **kwargs)
        
        torch.load = patched_load
        
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded: {model_path}")
            logger.info(f"Device: {self.device}")
        finally:
            # Restore original
            torch.load = original_load
    
    def _load_with_context(self, model_path):
        try:
            from ultralytics.nn.tasks import DetectionModel
            from ultralytics.nn.modules.head import Detect
            
            with torch.serialization.safe_globals([DetectionModel, Detect]):
                self.model = YOLO(model_path)
                self.model.to(self.device)
                logger.info(f"Model loaded: {model_path}")
                logger.info(f"Device: {self.device}")
        except Exception as e:
            raise RuntimeError(f"Safe globals failed: {e}")
    
    def detect(self, image, conf=0.4):
        
        if self.model is None:
            return {
                'anomaly_count': 0,
                'image_annotated': image.copy(),
                'cracks': [],
                'confidence': conf,
            }
        
        try:
            # Run inference
            results = self.model(image, conf=conf, device=self.device, verbose=False)
            result = results[0]
            
            # Get boxes
            boxes = result.boxes.cpu().numpy()
            
            # Counting detections
            anomaly_count = len(boxes)
            
            # Annotate image
            image_annotated = result.plot()
            
            # Extract crack information
            cracks = []
            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = box.conf[0]
                    
                    crack_info = {
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'area': int((x2 - x1) * (y2 - y1)),
                    }
                    cracks.append(crack_info)
                except:
                    pass
            
            return {
                'anomaly_count': anomaly_count,
                'image_annotated': image_annotated,
                'cracks': cracks,
                'confidence': conf,
            }
        
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {
                'anomaly_count': 0,
                'image_annotated': image.copy(),
                'cracks': [],
                'confidence': conf,
            }
    
    def detect_with_preprocessing(self, image, conf=0.4):
        # preprocessing
        preprocessed = self._preprocess(image)
        
        # Detect
        return self.detect(preprocessed, conf=conf)
    
    def _preprocess(self, image):
        try:
            # Increase contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except:
            return image
