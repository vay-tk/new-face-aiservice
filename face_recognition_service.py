import cv2
import numpy as np
import face_recognition
import base64
import io
from PIL import Image
import os
from typing import List, Dict, Any
import math


class FaceRecognitionService:
    def __init__(self):
        self.face_match_threshold = float(os.getenv("FACE_MATCH_THRESHOLD", 0.6))
        self.ear_threshold = float(os.getenv("LIVENESS_EAR_THRESHOLD", 0.22))
        self.head_movement_threshold = float(os.getenv("LIVENESS_HEAD_MOVEMENT_THRESHOLD", 10))
        self.blink_frame_check = int(os.getenv("BLINK_FRAME_CHECK", 8))
        self.head_movement_min_angle = float(os.getenv("HEAD_MOVEMENT_MIN_ANGLE", 5))
        
    def base64_to_opencv(self, base64_string: str) -> np.ndarray:
        """Convert base64 string to OpenCV image"""
        try:
            # Remove data:image/jpeg;base64, prefix if present
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)
            
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert PIL to OpenCV format (RGB to BGR)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
        except Exception as e:
            raise Exception(f"Failed to convert base64 to OpenCV image: {str(e)}")
    
    def extract_embedding(self, base64_image: str) -> Dict[str, Any]:
        """Extract face embedding from base64 image"""
        try:
            # Convert base64 to OpenCV image
            image = self.base64_to_opencv(base64_image)
            
            # Convert BGR to RGB for face_recognition library
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return {"success": False, "error": "No face detected in image"}
            
            if len(face_locations) > 1:
                return {"success": False, "error": "Multiple faces detected. Please ensure only one face is visible."}
            
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                return {"success": False, "error": "Failed to extract face features"}
            
            return {
                "success": True,
                "embedding": face_encodings[0].tolist(),
                "face_location": face_locations[0]
            }
            
        except Exception as e:
            return {"success": False, "error": f"Face embedding extraction failed: {str(e)}"}
    
    def calculate_ear(self, eye_landmarks: List) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        try:
            # Convert landmarks to numpy array
            eye = np.array(eye_landmarks)
            
            # Calculate distances
            A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance 1
            B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance 2
            C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
            
            # Calculate EAR
            ear = (A + B) / (2.0 * C)
            return ear
            
        except Exception:
            return 0.3  # Default EAR value if calculation fails
    
    def calculate_head_pose(self, face_landmarks: Dict) -> Dict[str, float]:
        """Calculate head pose angles from face landmarks"""
        try:
            # Get nose tip and nose bridge points
            nose_tip = face_landmarks['nose_tip'][2]  # Center of nose tip
            nose_bridge = face_landmarks['nose_bridge'][1]  # Center of nose bridge
            
            # Calculate basic head rotation indicator
            nose_x_offset = nose_tip[0] - nose_bridge[0]
            
            # Simple head pose estimation based on nose position
            # This is a simplified approach - in production, you might want more sophisticated pose estimation
            head_yaw = nose_x_offset * 0.5  # Simplified yaw calculation
            
            return {
                "yaw": head_yaw,
                "nose_x": nose_tip[0],
                "nose_bridge_x": nose_bridge[0],
                "offset": nose_x_offset
            }
            
        except Exception as e:
            return {"yaw": 0.0, "nose_x": 0, "nose_bridge_x": 0, "offset": 0}
    
    def detect_liveness(self, frames: List[str]) -> Dict[str, Any]:
        """Detect liveness using blink and head movement detection"""
        try:
            if len(frames) < self.blink_frame_check:
                return {
                    "success": False,
                    "isLive": False,
                    "score": 0.0,
                    "blinkDetected": False,
                    "headMovementDetected": False,
                    "details": {},
                    "error": f"Insufficient frames. Need at least {self.blink_frame_check} frames."
                }
            
            ear_values = []
            nose_positions = []
            frame_count = 0
            
            for frame_b64 in frames:
                try:
                    # Convert frame to OpenCV image
                    frame = self.base64_to_opencv(frame_b64)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect face landmarks
                    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
                    
                    if not face_landmarks_list:
                        continue  # Skip frames without detected faces
                    
                    face_landmarks = face_landmarks_list[0]
                    
                    # Calculate EAR for both eyes
                    left_eye = face_landmarks.get('left_eye', [])
                    right_eye = face_landmarks.get('right_eye', [])
                    
                    if left_eye and right_eye:
                        left_ear = self.calculate_ear(left_eye)
                        right_ear = self.calculate_ear(right_eye)
                        avg_ear = (left_ear + right_ear) / 2.0
                        ear_values.append(avg_ear)
                    
                    # Calculate head pose
                    head_pose = self.calculate_head_pose(face_landmarks)
                    nose_positions.append(head_pose["nose_x"])
                    
                    frame_count += 1
                    
                except Exception as e:
                    continue  # Skip problematic frames
            
            if frame_count < 3:
                return {
                    "success": False,
                    "isLive": False,
                    "score": 0.0,
                    "blinkDetected": False,
                    "headMovementDetected": False,
                    "details": {},
                    "error": "Insufficient valid frames with face detection"
                }
            
            # Analyze blink detection
            blink_detected = False
            blink_count = 0
            
            if len(ear_values) >= 3:
                # Look for EAR drops indicating blinks
                # More flexible blink detection - look for significant EAR drops
                for i in range(len(ear_values) - 1):
                    if ear_values[i] < self.ear_threshold:
                        blink_count += 1
                        # Skip next frame to avoid double counting
                        if i + 1 < len(ear_values):
                            i += 1
                
                # Also check for overall variation in EAR values (indicates blinking)
                if len(ear_values) > 0:
                    ear_std = (sum([(x - sum(ear_values)/len(ear_values))**2 for x in ear_values]) / len(ear_values))**0.5
                    if ear_std > 0.02:  # Standard deviation threshold for blink variation
                        blink_count += 1
                
                blink_detected = blink_count >= 2  # Require at least 2 blink indicators
            
            # Analyze head movement
            head_movement_detected = False
            head_movement_range = 0
            
            if len(nose_positions) >= 3:
                nose_min = min(nose_positions)
                nose_max = max(nose_positions)
                head_movement_range = nose_max - nose_min
                head_movement_detected = head_movement_range > self.head_movement_threshold
            
            # Calculate overall liveness score
            blink_score = 1.0 if blink_detected else 0.0
            movement_score = min(1.0, head_movement_range / (self.head_movement_threshold * 2))
            overall_score = (blink_score * 0.6) + (movement_score * 0.4)
            
            # Determine if person is live
            is_live = bool(blink_detected and head_movement_detected)
            
            return {
                "success": True,
                "isLive": bool(is_live),
                "score": overall_score,
                "blinkDetected": bool(blink_detected),
                "headMovementDetected": bool(head_movement_detected),
                "details": {
                    "blinkCount": blink_count,
                    "blinkScore": blink_score,
                    "headMovementRange": head_movement_range,
                    "headMovementScore": movement_score,
                    "framesProcessed": frame_count,
                    "avgEAR": sum(ear_values) / len(ear_values) if ear_values else 0,
                    "earValues": [float(x) for x in ear_values],
                    "nosePositions": [float(x) for x in nose_positions]
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "isLive": bool(False),
                "score": 0.0,
                "blinkDetected": bool(False),
                "headMovementDetected": bool(False),
                "details": {},
                "error": f"Liveness detection failed: {str(e)}"
            }
    
    def match_face(self, probe_image: str, user_embeddings: List[List[float]]) -> Dict[str, Any]:
        """Match probe image against user embeddings"""
        try:
            if not user_embeddings:
                return {
                    "success": False,
                    "isMatch": bool(False),
                    "confidence": 0.0,
                    "bestMatchIndex": -1,
                    "error": "No enrolled face embeddings found for this user"
                }
            
            # Extract embedding from probe image
            probe_result = self.extract_embedding(probe_image)
            
            if not probe_result["success"]:
                return {
                    "success": False,
                    "isMatch": False,
                    "confidence": 0.0,
                    "bestMatchIndex": -1,
                    "error": probe_result["error"]
                }
            
            probe_embedding = np.array(probe_result["embedding"])
            
            # Calculate distances to all enrolled embeddings
            distances = []
            for i, enrolled_embedding in enumerate(user_embeddings):
                enrolled_array = np.array(enrolled_embedding)
                distance = face_recognition.face_distance([enrolled_array], probe_embedding)[0]
                distances.append(distance)
            
            # Find best match
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            # Convert distance to confidence (lower distance = higher confidence)
            confidence = max(0.0, 1.0 - best_distance)
            
            # Determine if it's a match - be more strict for security
            is_match = bool(best_distance < self.face_match_threshold and confidence > 0.6)
            
            return {
                "success": True,
                "isMatch": bool(is_match),
                "confidence": confidence,
                "bestMatchIndex": int(best_match_index),
                "distance": float(best_distance),
                "allDistances": [float(d) for d in distances],
                "threshold": float(self.face_match_threshold),
                "minConfidence": 0.6
            }
            
        except Exception as e:
            return {
                "success": False,
                "isMatch": bool(False),
                "confidence": 0.0,
                "bestMatchIndex": -1,
                "error": f"Face matching failed: {str(e)}"
            }