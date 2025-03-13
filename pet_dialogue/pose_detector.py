#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Animal Pose Detector module using MediaPipe.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from loguru import logger
from typing import Dict, List, Any, Optional, Tuple

from pet_dialogue.utils import decode_image

class PoseDetector:
    """
    Animal pose detection using MediaPipe Animal Pose.
    
    Detects key skeletal points of animals (dogs/cats) from images or video frames.
    """
    
    # Key animal keypoints with their indices
    DOG_KEYPOINTS = {
        "nose": 0,
        "left_ear": 1,
        "right_ear": 2,
        "neck": 3,
        "withers": 4,  # Top of shoulder blades
        "tail_base": 5,
        "left_front_paw": 6,
        "right_front_paw": 7,
        "left_back_paw": 8,
        "right_back_paw": 9,
        "left_elbow": 10,
        "right_elbow": 11,
        "left_knee": 12,
        "right_knee": 13
    }
    
    # Same structure for cats (indices may vary based on actual model)
    CAT_KEYPOINTS = {
        "nose": 0,
        "left_ear": 1,
        "right_ear": 2,
        "neck": 3,
        "withers": 4,
        "tail_base": 5,
        "left_front_paw": 6,
        "right_front_paw": 7,
        "left_back_paw": 8,
        "right_back_paw": 9,
        "left_elbow": 10,
        "right_elbow": 11,
        "left_knee": 12,
        "right_knee": 13
    }
    
    def __init__(self):
        """Initialize pose detector with MediaPipe model."""
        # Load configuration
        self.model_complexity = int(os.getenv("MEDIAPIPE_MODEL_COMPLEXITY", 2))
        self.min_detection_confidence = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
        
        logger.info(f"Initializing PoseDetector with model_complexity={self.model_complexity}, "
                   f"min_detection_confidence={self.min_detection_confidence}")
        
        # Initialize MediaPipe Pose
        self.mp_animal_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize the pose detector
        # Note: MediaPipe doesn't officially support animal pose yet, so we're using human pose
        # with the intention to replace it with animal-specific models when available
        self.pose = self.mp_animal_pose.Pose(
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        logger.info("MediaPipe Pose model loaded")
    
    def process_frame(self, frame_data: bytes) -> Dict[str, Any]:
        """
        Process a single frame to detect animal pose.
        
        Args:
            frame_data: Raw image data in bytes
            
        Returns:
            Dictionary containing detection results with keypoints and confidence scores
        """
        # Decode the image
        frame = decode_image(frame_data)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        # No pose detected
        if not results.pose_landmarks:
            logger.debug("No pose detected in frame")
            return {}
        
        # Extract keypoints and normalize to [0,1] range
        keypoints = []
        h, w, _ = frame.shape
        
        for landmark in results.pose_landmarks.landmark:
            keypoints.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
        
        # For demonstration, we'll map human pose keypoints to animal keypoints
        # This is a placeholder - ideally we'd use a specialized animal pose model
        animal_keypoints = self._map_human_to_animal_keypoints(keypoints)
        
        return {
            "keypoints": animal_keypoints,
            "confidence": self._calculate_detection_confidence(keypoints),
            "animal_type": self._detect_animal_type(keypoints)  # Dog or cat detection
        }
    
    def visualize_pose(self, frame_data: bytes, results: Dict[str, Any]) -> bytes:
        """
        Draw pose keypoints and connections on the frame.
        
        Args:
            frame_data: Raw image data in bytes
            results: Pose detection results
            
        Returns:
            Annotated frame as bytes
        """
        frame = decode_image(frame_data)
        
        # No pose detected
        if not results or not results.get("keypoints"):
            return frame
        
        # Draw keypoints and connections
        for i, keypoint in enumerate(results["keypoints"]):
            # Only draw visible keypoints
            if keypoint["visibility"] > 0.5:
                cx, cy = int(keypoint["x"] * frame.shape[1]), int(keypoint["y"] * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"{i}", (cx + 10, cy + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw connections between keypoints (simplified for demonstration)
        # In a real implementation, you would define the proper skeleton connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 3),  # Head
            (3, 4), (4, 5),  # Spine
            (4, 6), (4, 7),  # Shoulders to front paws
            (5, 8), (5, 9),  # Hips to back paws
            (6, 10), (7, 11),  # Front leg joints
            (8, 12), (9, 13)   # Back leg joints
        ]
        
        for conn in connections:
            if (results["keypoints"][conn[0]]["visibility"] > 0.5 and 
                results["keypoints"][conn[1]]["visibility"] > 0.5):
                pt1 = (int(results["keypoints"][conn[0]]["x"] * frame.shape[1]), 
                       int(results["keypoints"][conn[0]]["y"] * frame.shape[0]))
                pt2 = (int(results["keypoints"][conn[1]]["x"] * frame.shape[1]), 
                       int(results["keypoints"][conn[1]]["y"] * frame.shape[0]))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        
        # Add animal type and confidence
        cv2.putText(frame, f"Animal: {results.get('animal_type', 'Unknown')}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Confidence: {results.get('confidence', 0):.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Encode frame back to bytes
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            logger.error("Failed to encode visualized frame")
            return frame_data
        
        return buffer.tobytes()
    
    def _map_human_to_animal_keypoints(self, human_keypoints: List[Dict]) -> List[Dict]:
        """
        Map human pose keypoints to animal pose keypoints.
        This is a placeholder for demonstration - ideally we'd use animal-specific models.
        
        Args:
            human_keypoints: List of human pose keypoints
            
        Returns:
            List of animal pose keypoints
        """
        # Very simplistic mapping for demonstration
        # In reality, this would be much more sophisticated
        mapping = {
            0: 0,    # nose to nose
            7: 1,    # left ear to left ear
            8: 2,    # right ear to right ear
            9: 3,    # mouth to neck (approximation)
            11: 4,   # left shoulder to withers
            23: 5,   # hip to tail base (approximation)
            15: 6,   # left wrist to left front paw
            16: 7,   # right wrist to right front paw
            27: 8,   # left ankle to left back paw
            28: 9,   # right ankle to right back paw
            13: 10,  # left elbow to left elbow
            14: 11,  # right elbow to right elbow
            25: 12,  # left knee to left knee
            26: 13   # right knee to right knee
        }
        
        animal_keypoints = []
        for i in range(14):  # Number of animal keypoints
            if i in mapping.values():
                # Find the human keypoint index that maps to this animal keypoint
                human_idx = [k for k, v in mapping.items() if v == i][0]
                # Check if the human keypoint is within range
                if human_idx < len(human_keypoints):
                    animal_keypoints.append(human_keypoints[human_idx])
                else:
                    # Fallback for missing keypoints
                    animal_keypoints.append({"x": 0, "y": 0, "z": 0, "visibility": 0})
            else:
                # Keypoint not mapped
                animal_keypoints.append({"x": 0, "y": 0, "z": 0, "visibility": 0})
        
        return animal_keypoints
    
    def _calculate_detection_confidence(self, keypoints: List[Dict]) -> float:
        """
        Calculate overall detection confidence based on keypoint visibility.
        
        Args:
            keypoints: List of detected keypoints
            
        Returns:
            Overall confidence score (0-1)
        """
        if not keypoints:
            return 0.0
        
        # Average visibility of key points as a confidence metric
        visibilities = [kp["visibility"] for kp in keypoints if "visibility" in kp]
        if not visibilities:
            return 0.0
        
        return sum(visibilities) / len(visibilities)
    
    def _detect_animal_type(self, keypoints: List[Dict]) -> str:
        """
        Determine if the detected animal is a dog or cat based on pose characteristics.
        This is a placeholder - in a real implementation, we'd use a classifier.
        
        Args:
            keypoints: List of detected keypoints
            
        Returns:
            "Dog" or "Cat" classification
        """
        # Placeholder logic - in reality, this would be a trained classifier
        # For now, we'll return "Dog" as default
        return "Dog" 