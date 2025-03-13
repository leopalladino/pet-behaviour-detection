#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Behavior classifier for mapping animal poses to emotional/behavioral states.
"""

import os
import math
import numpy as np
from loguru import logger
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from pet_dialogue.utils import load_json_data, get_project_root

class BehaviorClassifier:
    """
    Maps detected animal poses to predefined emotional/behavioral states.
    
    Uses a rule-based approach to classify poses into states like:
    - Excitement
    - Anxiety
    - Curiosity
    - Playfulness
    - Relaxation
    - Alert
    - etc.
    """
    
    # Major behavioral states
    BEHAVIORS = [
        "excited",
        "anxious",
        "curious",
        "playful",
        "relaxed",
        "alert",
        "tired",
        "hungry",
        "affectionate",
        "defensive",
        "angry",
        "aggressive"
    ]
    
    def __init__(self):
        """Initialize the behavior classifier."""
        logger.info("Initializing BehaviorClassifier")
        
        # Load behavior rules
        self.rules = self._load_behavior_rules()
        
        # Configure confidence threshold - lower for angry/aggressive behaviors
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
        self.angry_confidence_threshold = 0.4  # Lower threshold for angry behaviors
        
        logger.info(f"Loaded {len(self.rules)} behavior classification rules")
    
    def classify(self, pose_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a detected pose into behavioral states.
        
        Args:
            pose_results: Detection results from PoseDetector
            
        Returns:
            Dictionary with behavior classification results
        """
        if not pose_results or not pose_results.get("keypoints"):
            logger.debug("No valid pose data for behavior classification")
            return {"state": "unknown", "confidence": 0.0, "details": {}}
        
        # Extract key features from the pose
        features = self._extract_pose_features(pose_results)
        logger.debug(f"Extracted features: {features}")
        
        # Calculate behavior scores
        behavior_scores = self._calculate_behavior_scores(features, pose_results["animal_type"])
        logger.debug(f"Behavior scores: {behavior_scores}")
        
        # Get the top behavior
        top_behavior = max(behavior_scores.items(), key=lambda x: x[1])
        logger.debug(f"Top behavior: {top_behavior[0]} with score {top_behavior[1]}")
        
        # Always return the behavior with highest confidence
        return {
            "state": top_behavior[0],
            "confidence": top_behavior[1],
            "details": behavior_scores
        }
    
    def _load_behavior_rules(self) -> Dict:
        """
        Load behavior classification rules from JSON file.
        
        If file doesn't exist, use default rules.
        
        Returns:
            Dictionary of behavior rules
        """
        rules_path = get_project_root() / "pet_dialogue" / "data" / "behavior_rules.json"
        
        if rules_path.exists():
            try:
                return load_json_data(rules_path)
            except Exception as e:
                logger.error(f"Failed to load behavior rules: {e}")
        
        # Default rules if file doesn't exist or can't be loaded
        return self._default_behavior_rules()
    
    def _default_behavior_rules(self) -> Dict:
        """
        Define default behavior classification rules.
        
        Returns:
            Dictionary of behavior rules
        """
        return {
            "dog": {
                "excited": {
                    "tail_height": "high",
                    "tail_wag": "fast",
                    "ear_position": "forward",
                    "body_posture": "bouncy"
                },
                "anxious": {
                    "tail_position": "tucked",
                    "ear_position": "back",
                    "body_posture": "lowered"
                },
                "curious": {
                    "head_tilt": "high",
                    "ear_position": "forward",
                    "body_posture": "attentive"
                },
                "playful": {
                    "front_stance": "lowered",
                    "tail_wag": "medium",
                    "body_posture": "bouncy"
                },
                "relaxed": {
                    "body_posture": "stretched",
                    "tail_wag": "slow",
                    "ear_position": "neutral"
                },
                "alert": {
                    "ear_position": "forward",
                    "body_posture": "rigid",
                    "tail_position": "straight"
                },
                "angry": {
                    "ear_position": "forward",
                    "body_posture": "tense",
                    "tail_position": "high",
                    "head_position": "forward",
                    "body_tension": "tense"
                },
                "aggressive": {
                    "ear_position": "forward",
                    "body_posture": "tense",
                    "tail_position": "high",
                    "head_position": "forward",
                    "body_tension": "tense",
                    "front_stance": "high"
                }
            },
            "cat": {
                "excited": {
                    "tail_position": "upright",
                    "tail_movement": "twitching",
                    "ear_position": "forward"
                },
                "anxious": {
                    "body_posture": "arched",
                    "tail_position": "tucked",
                    "ear_position": "flat"
                },
                "curious": {
                    "ear_position": "forward",
                    "tail_position": "upright",
                    "head_movement": "scanning"
                },
                "playful": {
                    "body_posture": "crouched",
                    "tail_movement": "swishing",
                    "rear_position": "wiggling"
                },
                "relaxed": {
                    "body_posture": "stretched",
                    "tail_position": "curled",
                    "ear_position": "neutral"
                },
                "alert": {
                    "ear_position": "forward",
                    "tail_position": "stiff",
                    "body_posture": "tense"
                },
                "angry": {
                    "body_posture": "arched",
                    "tail_position": "low",
                    "tail_movement": "swishing",
                    "ear_position": "back"
                },
                "aggressive": {
                    "body_posture": "low",
                    "ear_position": "flat",
                    "tail_position": "low",
                    "tail_movement": "rapid"
                }
            }
        }
    
    def _extract_pose_features(self, pose_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract high-level features from raw pose keypoints.
        
        Args:
            pose_results: Detection results from PoseDetector
            
        Returns:
            Dictionary of extracted features
        """
        keypoints = pose_results.get("keypoints", [])
        
        if not keypoints or len(keypoints) < 14:
            logger.debug("No valid pose data for behavior classification")
            return {}
        
        # Extract coordinates for key points (normalized)
        nose = (keypoints[0]["x"], keypoints[0]["y"])
        left_ear = (keypoints[1]["x"], keypoints[1]["y"])
        right_ear = (keypoints[2]["x"], keypoints[2]["y"])
        neck = (keypoints[3]["x"], keypoints[3]["y"])
        withers = (keypoints[4]["x"], keypoints[4]["y"])
        tail_base = (keypoints[5]["x"], keypoints[5]["y"])
        left_shoulder = (keypoints[6]["x"], keypoints[6]["y"])
        right_shoulder = (keypoints[7]["x"], keypoints[7]["y"])
        left_elbow = (keypoints[8]["x"], keypoints[8]["y"])
        right_elbow = (keypoints[9]["x"], keypoints[9]["y"])
        left_wrist = (keypoints[10]["x"], keypoints[10]["y"])
        right_wrist = (keypoints[11]["x"], keypoints[11]["y"])
        left_hip = (keypoints[12]["x"], keypoints[12]["y"])
        right_hip = (keypoints[13]["x"], keypoints[13]["y"])
        
        logger.debug("\nExtracting pose features:")
        logger.debug(f"Nose: {nose}")
        logger.debug(f"Left ear: {left_ear}")
        logger.debug(f"Right ear: {right_ear}")
        logger.debug(f"Neck: {neck}")
        logger.debug(f"Withers: {withers}")
        logger.debug(f"Tail base: {tail_base}")
        
        # Calculate key feature metrics
        
        # Head tilt (angle between horizontal and line connecting ears)
        head_tilt = self._calculate_angle(left_ear, right_ear)
        logger.debug(f"Head tilt angle: {head_tilt:.2f} degrees")
        
        # Ear position (distance from ears to neck, normalized by head size)
        ear_height = (left_ear[1] + right_ear[1]) / 2 - neck[1]
        head_size = self._calculate_distance(left_ear, right_ear)
        normalized_ear_height = ear_height / head_size if head_size > 0 else 0
        logger.debug(f"Ear height: {ear_height:.3f}, Head size: {head_size:.3f}, Normalized ear height: {normalized_ear_height:.3f}")
        
        # Body posture (angle of spine) - fixed calculation to use absolute angle
        spine_angle = abs(self._calculate_angle(neck, tail_base))
        # Normalize spine angle to be between 0 and 180 degrees
        if spine_angle < 0:
            spine_angle += 180
        logger.debug(f"Spine angle: {spine_angle:.2f} degrees")
        
        # Body height (vertical distance from withers to ground, normalized)
        body_height = withers[1]
        logger.debug(f"Body height: {body_height:.3f}")
        
        # Tail position (if visible)
        tail_position = tail_base[1]  # Y coordinate (height)
        logger.debug(f"Tail position: {tail_position:.3f}")
        
        # Front leg stance
        left_front_angle = self._calculate_angle(left_shoulder, left_elbow)
        right_front_angle = self._calculate_angle(right_shoulder, right_elbow)
        front_stance = (left_front_angle + right_front_angle) / 2
        logger.debug(f"Front leg stance: {front_stance:.2f} degrees (left: {left_front_angle:.2f}, right: {right_front_angle:.2f})")
        
        # Back leg stance
        left_back_angle = self._calculate_angle(left_hip, left_wrist)
        right_back_angle = self._calculate_angle(right_hip, right_wrist)
        back_stance = (left_back_angle + right_back_angle) / 2
        logger.debug(f"Back leg stance: {back_stance:.2f} degrees (left: {left_back_angle:.2f}, right: {right_back_angle:.2f})")
        
        # Body tension (based on shoulder and hip angles)
        shoulder_width = self._calculate_distance(left_shoulder, right_shoulder)
        hip_width = self._calculate_distance(left_hip, right_hip)
        body_tension = abs(shoulder_width - hip_width) / max(shoulder_width, hip_width)
        logger.debug(f"Body tension: {body_tension:.3f} (shoulder width: {shoulder_width:.3f}, hip width: {hip_width:.3f})")
        
        # Head position relative to body
        head_position = nose[1] - neck[1]  # Negative means head is up
        logger.debug(f"Head position: {head_position:.3f}")
        
        features = {
            "head_tilt": head_tilt,
            "ear_height": normalized_ear_height,
            "ear_position": normalized_ear_height,  # Add ear_position as alias for ear_height
            "spine_angle": spine_angle,
            "body_height": body_height,
            "tail_position": tail_position,
            "front_stance": front_stance,
            "back_stance": back_stance,
            "body_tension": body_tension,
            "head_position": head_position,
            "head_size": head_size
        }
        
        logger.debug("\nFinal extracted features:")
        for feature, value in features.items():
            logger.debug(f"{feature}: {value:.3f}")
        
        return features
    
    def _calculate_behavior_scores(self, features: Dict[str, float], animal_type: str) -> Dict[str, float]:
        """
        Calculate behavior scores based on extracted features.
        
        Args:
            features: Dictionary of extracted features
            animal_type: Type of animal ("dog" or "cat")
            
        Returns:
            Dictionary mapping behavior names to confidence scores
        """
        scores = {}
        rules = self.rules.get(animal_type, {})
        
        logger.debug(f"\nCalculating behavior scores for {animal_type}")
        logger.debug(f"Available rules: {list(rules.keys())}")
        
        for behavior, rule in rules.items():
            score = 0.0
            total_weight = 0
            weighted_matches = 0
            
            logger.debug(f"\nCalculating score for behavior: {behavior}")
            logger.debug(f"Rule requirements: {rule}")
            
            # Handle new rule format with features and min_confidence
            if isinstance(rule, dict) and "features" in rule:
                feature_rules = rule["features"]
                min_confidence = rule.get("min_confidence", 0.5)
                
                for feature, feature_rule in feature_rules.items():
                    if feature not in features:
                        logger.debug(f"Feature {feature} not found in extracted features")
                        continue
                        
                    feature_value = features[feature]
                    weight = feature_rule.get("weight", 1.0)
                    total_weight += weight
                    
                    logger.debug(f"Feature {feature}: value={feature_value:.3f}, rule={feature_rule}")
                    
                    # Handle different rule types
                    if "threshold" in feature_rule:
                        # Threshold comparison
                        threshold = feature_rule["threshold"]
                        comparison = feature_rule.get("comparison", "less")
                        
                        if comparison == "less" and feature_value < threshold:
                            weighted_matches += weight
                            logger.debug(f"Threshold match: {feature} < {threshold}")
                        elif comparison == "greater" and feature_value > threshold:
                            weighted_matches += weight
                            logger.debug(f"Threshold match: {feature} > {threshold}")
                        else:
                            logger.debug(f"Threshold mismatch: {feature} {comparison} {threshold}")
                            
                    elif "min" in feature_rule and "max" in feature_rule:
                        # Range check
                        min_val = feature_rule["min"]
                        max_val = feature_rule["max"]
                        
                        if min_val <= feature_value <= max_val:
                            weighted_matches += weight
                            logger.debug(f"Range match: {min_val} <= {feature_value} <= {max_val}")
                        else:
                            logger.debug(f"Range mismatch: {feature_value} not in [{min_val}, {max_val}]")
                            
                    elif "reference" in feature_rule:
                        # Reference angle with deviation
                        reference = feature_rule["reference"]
                        deviation = feature_rule.get("deviation", 30)
                        min_deviation = feature_rule.get("min_deviation", 10)
                        
                        # Calculate angle difference
                        angle_diff = abs(feature_value - reference)
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff
                            
                        if angle_diff <= deviation and angle_diff >= min_deviation:
                            weighted_matches += weight
                            logger.debug(f"Angle match: {angle_diff}° within [{min_deviation}°, {deviation}°]")
                        else:
                            logger.debug(f"Angle mismatch: {angle_diff}° not in [{min_deviation}°, {deviation}°]")
                    
                    else:
                        logger.debug(f"Unknown rule type for feature {feature}")
                        continue
                
                # Calculate final score
                if total_weight > 0:
                    score = weighted_matches / total_weight
                    logger.debug(f"Raw score: {score:.3f} (matches: {weighted_matches}, total weight: {total_weight})")
                    
                    # Apply minimum confidence threshold
                    if score < min_confidence:
                        score = 0.0
                        logger.debug(f"Score below minimum confidence threshold ({min_confidence})")
                else:
                    logger.debug("No rules matched")
            
            # Handle old rule format (for backward compatibility)
            else:
                for feature, expected_value in rule.items():
                    if feature in features:
                        feature_value = features[feature]
                        logger.debug(f"Feature {feature}: value={feature_value:.3f}, expected={expected_value}")
                        
                        # Convert feature value to categorical
                        if feature == "head_tilt":
                            if feature_value > 30:
                                actual_value = "high"
                            elif feature_value < -30:
                                actual_value = "low"
                            else:
                                actual_value = "neutral"
                        elif feature == "ear_height":
                            if feature_value > 0.2:
                                actual_value = "high"
                            elif feature_value < -0.2:
                                actual_value = "low"
                            else:
                                actual_value = "neutral"
                        elif feature == "spine_angle":
                            if feature_value > 30:
                                actual_value = "high"
                            elif feature_value < -30:
                                actual_value = "low"
                            else:
                                actual_value = "neutral"
                        elif feature == "body_height":
                            if feature_value > 0.8:
                                actual_value = "high"
                            elif feature_value < 0.4:
                                actual_value = "low"
                            else:
                                actual_value = "medium"
                        elif feature == "tail_position":
                            if feature_value > 0.7:
                                actual_value = "high"
                            elif feature_value < 0.4:
                                actual_value = "low"
                            else:
                                actual_value = "medium"
                        elif feature == "front_stance":
                            if feature_value > 20:
                                actual_value = "high"
                            elif feature_value < -20:
                                actual_value = "low"
                            else:
                                actual_value = "neutral"
                        elif feature == "back_stance":
                            if feature_value > 30:
                                actual_value = "high"
                            elif feature_value < -30:
                                actual_value = "low"
                            else:
                                actual_value = "neutral"
                        elif feature == "body_tension":
                            if feature_value > 0.25:
                                actual_value = "tense"
                            else:
                                actual_value = "relaxed"
                        elif feature == "head_position":
                            if feature_value > 0.05:
                                actual_value = "high"
                            elif feature_value < -0.05:
                                actual_value = "low"
                            else:
                                actual_value = "neutral"
                        else:
                            logger.debug(f"Unknown feature type: {feature}")
                            continue
                        
                        logger.debug(f"Converted {feature} value to: {actual_value}")
                        
                        if actual_value == expected_value:
                            weighted_matches += 1
                            total_weight += 1
                            logger.debug(f"Rule match! Current matches: {weighted_matches}, total weight: {total_weight}")
                        else:
                            logger.debug(f"Rule mismatch: got {actual_value}, expected {expected_value}")
                
                if total_weight > 0:
                    score = weighted_matches / total_weight
                    logger.debug(f"Final score for {behavior}: {score:.3f}")
                else:
                    logger.debug(f"No rules matched for {behavior}")
            
            scores[behavior] = score
        
        logger.debug("\nFinal behavior scores:")
        for behavior, score in scores.items():
            logger.debug(f"{behavior}: {score:.3f}")
        
        return scores
    
    def _calculate_angle(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate angle between two points relative to horizontal."""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.degrees(math.atan2(dy, dx))
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx*dx + dy*dy)