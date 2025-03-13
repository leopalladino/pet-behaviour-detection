#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Processor module for Pet Dialogue.
Handles pre-recorded videos, analyzes frames, and generates output with overlays.
"""

import os
import cv2
import numpy as np
import tempfile
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Generator
from loguru import logger
from tqdm import tqdm
import math
import random

from pet_dialogue.pose_detector import PoseDetector
from pet_dialogue.behavior_classifier import BehaviorClassifier
from pet_dialogue.dialogue_generator import DialogueGenerator
from pet_dialogue.utils import encode_image, decode_image

class VideoProcessor:
    """
    Processes pre-recorded videos of animals to detect poses, classify behaviors,
    and generate overlaid dialogue based on the animal's perceived state.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the video processor.
        
        Args:
            output_dir: Directory to save processed videos
        """
        logger.info("Initializing VideoProcessor")
        
        # Initialize components
        self.pose_detector = PoseDetector()
        self.behavior_classifier = BehaviorClassifier()
        self.dialogue_generator = DialogueGenerator()
        
        # Set output directory
        self.output_dir = output_dir or "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Processing settings
        self.dialogue_duration_seconds = 3.0  # Display speech bubbles for exactly 3 seconds
        self.dialogue_cooldown_seconds = 5  # 5 second cooldown between dialogues
        self.min_dialogue_interval = 5  # Minimum seconds between dialogues
        self.max_dialogue_interval = 10  # Maximum seconds between dialogues
        
        # Speech bubble settings
        self.speech_bubble_bg = (255, 255, 255)  # white
        self.speech_bubble_text_color = (255, 0, 0)  # blue - in BGR format (OpenCV uses BGR)
        self.speech_bubble_border_color = (100, 100, 100)  # gray
        self.speech_bubble_opacity = 0.85  # 85% opaque
        self.speech_bubble_font_scale = 0.8  # Increased font size
        self.speech_bubble_thickness = 2  # Increased text thickness
        
        # Fixed position speech bubble (centered near top)
        self.fixed_bubble_position = True  # Use fixed position instead of following head
        
        # Caption settings
        self.caption_bg = (0, 0, 0)  # black
        self.caption_text_color = (255, 255, 255)  # white
        self.caption_opacity = 0.7  # 70% opaque
        
        logger.info("VideoProcessor initialized")
    
    async def process_video(self, 
                     video_path: str, 
                     animal_type: str = "auto", 
                     personality: str = "auto", 
                     display_mode: str = "speech_bubble",
                     forced_state: Optional[str] = None) -> str:
        """
        Process a video file, analyze frames, and generate overlaid dialogue.
        
        Args:
            video_path: Path to input video file
            animal_type: Type of animal ("dog", "cat", or "auto")
            personality: Personality trait for dialogue generation
            display_mode: Display mode for text ("caption", "speech_bubble", or "both")
            forced_state: Override detected state for testing (e.g., "angry", "playful")
            
        Returns:
            Path to processed output video
        """
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Settings - Animal: {animal_type}, Personality: {personality}, Display: {display_mode}")
        if forced_state:
            logger.info(f"Forcing behavior state: {forced_state}")
        
        # Validate input file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Calculate dialogue display duration in frames based on fps
        dialogue_frames = int(self.dialogue_duration_seconds * fps)
        logger.info(f"Speech bubbles will display for {self.dialogue_duration_seconds} seconds ({dialogue_frames} frames)")
        
        # Create output video file
        output_path = self._get_output_path(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec for better compatibility
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize processing variables
        frame_idx = 0
        current_dialogue = ""
        dialogue_frames_left = 0
        prev_behavior = None
        next_dialogue_frame = 0  # Track when to show next dialogue
        
        # Calculate frames for cooldown
        cooldown_frames = int(self.dialogue_cooldown_seconds * fps)
        cooldown_counter = 0
        
        # Set initial dialogue time (random between 0.5 and 2 seconds)
        initial_delay = random.uniform(0.5, 2.0)
        next_dialogue_frame = int(initial_delay * fps)
        
        # Save initial head position for reference
        head_position = None
        
        # Process video frames
        with tqdm(total=total_frames, desc="Processing video") as progress_bar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # Skip frames for efficiency (process every 5th frame)
                if frame_idx % 5 == 0 or dialogue_frames_left <= 0:
                    # Convert frame to bytes for processing
                    frame_bytes = encode_image(frame)
                    
                    # Detect pose
                    pose_results = self.pose_detector.process_frame(frame_bytes)
                    
                    if pose_results and pose_results.get("keypoints"):
                        # Save head position for speech bubble if not already stored
                        if head_position is None:
                            # Extract head position from the first good detection
                            bubble_pos = self._calculate_bubble_position(frame, pose_results)
                            head_position = (bubble_pos[0], bubble_pos[1])
                        
                        # Set animal type if auto-detect
                        if animal_type == "auto":
                            detected_animal = pose_results.get("animal_type", "dog").lower()
                        else:
                            # Override with user-specified animal type
                            pose_results["animal_type"] = animal_type
                            detected_animal = animal_type.lower()
                        
                        # Classify behavior
                        behavior = self.behavior_classifier.classify(pose_results)
                        
                        # Override behavior state if forced_state is provided
                        if forced_state:
                            # Store original confidence and details
                            confidence = behavior.get("confidence", 0.0)
                            details = behavior.get("details", {})
                            
                            # Override with forced state but maintain other properties
                            behavior = {
                                "state": forced_state,
                                "confidence": max(confidence, 0.8),  # Ensure high confidence
                                "details": details
                            }
                            logger.debug(f"Overriding detected state with forced state: {forced_state}")
                        
                        # Generate new dialogue when it's time or behavior changes significantly
                        behavior_changed = prev_behavior != behavior.get("state")
                        if (frame_idx >= next_dialogue_frame or behavior_changed) and cooldown_counter <= 0:
                            # Generate dialogue
                            dialogue = await self.dialogue_generator.generate(behavior, personality)
                            
                            if dialogue:
                                # Set new dialogue and reset persistence
                                current_dialogue = dialogue
                                dialogue_frames_left = dialogue_frames  # Set exact 3-second duration
                                prev_behavior = behavior.get("state")
                                
                                # Set next dialogue time (random between 5-10 seconds after current dialogue ends)
                                next_delay = random.uniform(self.min_dialogue_interval, self.max_dialogue_interval)
                                next_dialogue_frame = frame_idx + dialogue_frames + int(next_delay * fps)
                                
                                # Reset cooldown counter
                                cooldown_counter = cooldown_frames
                    else:
                        # No pose detected, decrement dialogue persistence
                        dialogue_frames_left = max(0, dialogue_frames_left - 1)
                else:
                    # Skipped detection frame, just decrement dialogue persistence
                    dialogue_frames_left = max(0, dialogue_frames_left - 1)
                
                # Decrement cooldown counter
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                
                # If we have dialogue to display
                if current_dialogue and dialogue_frames_left > 0:
                    # Draw the dialogue
                    if display_mode == "speech_bubble" or display_mode == "both":
                        frame = self._draw_speech_bubble(frame, current_dialogue, pose_results, head_position)
                    
                    if display_mode == "caption" or display_mode == "both":
                        frame = self._draw_caption(frame, current_dialogue)
                
                # Add watermark
                frame = self._add_watermark(frame)
                
                # Write frame to output video
                out.write(frame)
                
                # Update progress
                progress_bar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        
        # Convert to web-compatible format if needed
        output_path = self._ensure_web_compatible(output_path)
        
        logger.info(f"Video processing complete. Output: {output_path}")
        return output_path
    
    def preprocess_video(self, video_path: str) -> str:
        """
        Preprocess a video to improve quality for analysis.
        
        - Removes static/silent frames
        - Stabilizes footage if needed
        - Normalizes lighting
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Path to preprocessed video
        """
        logger.info(f"Preprocessing video: {video_path}")
        
        # TODO: Implement video preprocessing
        # - Frame stabilization
        # - Remove static sections
        # - Lighting normalization
        
        # For now, just return the original video path
        return video_path
    
    def _draw_speech_bubble(self, frame: np.ndarray, text: str, pose_results: Dict[str, Any], fixed_position=None) -> np.ndarray:
        """
        Draw a classic speech bubble with text.
        
        Args:
            frame: Video frame
            text: Text to display
            pose_results: Pose detection results
            fixed_position: Optional fixed position for speech bubble
            
        Returns:
            Frame with speech bubble
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Use fixed position if provided or setting enabled
        if fixed_position and self.fixed_bubble_position:
            bubble_x, bubble_y = fixed_position
            head_x, head_y = frame_width // 2, frame_height // 3  # Default head position
        elif pose_results and pose_results.get("keypoints"):
            # Calculate bubble position based on head position
            bubble_pos = self._calculate_bubble_position(frame, pose_results)
            bubble_x, bubble_y = bubble_pos[0], bubble_pos[1]
            head_x, head_y = bubble_pos[2], bubble_pos[3]
        else:
            # Default positioning
            bubble_x, bubble_y = frame_width // 4, frame_height // 4
            head_x, head_y = frame_width // 2, frame_height // 3
        
        # Prepare text - Use non-italic font with larger size and bold
        font = cv2.FONT_HERSHEY_SIMPLEX  # Non-italic font
        font_scale = self.speech_bubble_font_scale  # Increased font size
        thickness = self.speech_bubble_thickness  # Increased text thickness
        
        # Split text into lines if too long
        words = text.split()
        lines = []
        current_line = []
        
        # Target around 25 characters per line
        for word in words:
            if current_line and len(' '.join(current_line + [word])) > 25:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        if not lines:
            lines = [text]
        
        # Calculate text metrics
        line_heights = []
        line_widths = []
        
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            line_heights.append(text_size[1])
            line_widths.append(text_size[0])
        
        # Calculate bubble size - make it more compact
        padding = 25  # Increased padding
        line_spacing = 12  # Increased line spacing
        
        bubble_width = max(line_widths) + padding * 2
        bubble_height = sum(line_heights) + padding * 2 + line_spacing * (len(lines) - 1)
        
        # Ensure bubble is within frame boundaries
        x = max(10, min(bubble_x, frame_width - bubble_width - 10))
        y = max(10, min(bubble_y, frame_height - bubble_height - 10))
        
        # Create bubble overlay
        overlay = frame.copy()
        
        # Draw classic speech bubble shape
        # Main oval
        cv2.ellipse(overlay, 
                   (x + bubble_width//2, y + bubble_height//2),
                   (bubble_width//2, bubble_height//2),
                   0, 0, 360, self.speech_bubble_bg, -1)
        cv2.ellipse(overlay, 
                   (x + bubble_width//2, y + bubble_height//2),
                   (bubble_width//2, bubble_height//2),
                   0, 0, 360, self.speech_bubble_border_color, 2)
        
        # Draw tail if not fixed position
        if not self.fixed_bubble_position:
            # Calculate tail points
            tail_points = self._calculate_bubble_tail(x, y, bubble_width, bubble_height, head_x, head_y)
            
            # Draw tail
            cv2.fillPoly(overlay, [np.array(tail_points)], self.speech_bubble_bg)
            cv2.polylines(overlay, [np.array(tail_points)], True, self.speech_bubble_border_color, 2)
        
        # Blend overlay with original frame
        frame = cv2.addWeighted(overlay, self.speech_bubble_opacity, frame, 
                               1 - self.speech_bubble_opacity, 0)
        
        # Add text lines
        y_offset = y + padding + line_heights[0]
        for i, line in enumerate(lines):
            if i > 0:
                y_offset += line_heights[i-1] + line_spacing
            
            text_x = x + (bubble_width - line_widths[i]) // 2  # Center each line
            
            # Add text with blue color and bold style
            cv2.putText(frame, line, (text_x, y_offset), font, font_scale, 
                       self.speech_bubble_text_color, thickness)
        
        return frame
    
    def _draw_caption(self, frame: np.ndarray, text: str) -> np.ndarray:
        """
        Draw a caption at the bottom of the frame.
        
        Args:
            frame: Video frame
            text: Text to display
            
        Returns:
            Frame with caption
        """
        # Prepare text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Calculate caption position
        padding = 20
        caption_width = text_size[0] + padding * 2
        caption_height = text_size[1] + padding * 2
        
        caption_x = (frame.shape[1] - caption_width) // 2
        caption_y = frame.shape[0] - caption_height - 20  # 20px from bottom
        
        # Create caption overlay
        overlay = frame.copy()
        
        # Draw caption background
        cv2.rectangle(overlay, 
                     (caption_x, caption_y), 
                     (caption_x + caption_width, caption_y + caption_height), 
                     self.caption_bg, -1)
        
        # Blend overlay with original frame
        frame = cv2.addWeighted(overlay, self.caption_opacity, frame, 
                               1 - self.caption_opacity, 0)
        
        # Add text
        text_x = caption_x + padding
        text_y = caption_y + padding + text_size[1]
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                   self.caption_text_color, thickness)
        
        return frame
    
    def _add_watermark(self, frame: np.ndarray) -> np.ndarray:
        """
        Add a watermark to the frame.
        
        Args:
            frame: Video frame
            
        Returns:
            Frame with watermark
        """
        watermark_text = "For entertainment only"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Calculate position (bottom right corner)
        text_size = cv2.getTextSize(watermark_text, font, font_scale, thickness)[0]
        x = frame.shape[1] - text_size[0] - 10
        y = frame.shape[0] - 10
        
        # Add shadow for readability
        cv2.putText(frame, watermark_text, (x+1, y+1), font, font_scale, (0, 0, 0), thickness)
        
        # Add watermark text
        cv2.putText(frame, watermark_text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def _calculate_bubble_position(self, frame: np.ndarray, pose_results: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Calculate the optimal position for the speech bubble.
        
        Args:
            frame: Video frame
            pose_results: Pose detection results
            
        Returns:
            Tuple of (bubble_x, bubble_y, head_x, head_y)
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Default position (top left)
        default_pos = (20, 20, frame_width // 2, frame_height // 2)
        
        if not pose_results or not pose_results.get("keypoints"):
            return default_pos
        
        keypoints = pose_results.get("keypoints", [])
        
        if len(keypoints) < 3:
            return default_pos
        
        # Get head position (average of nose and ears)
        nose = keypoints[0]
        left_ear = keypoints[1]
        right_ear = keypoints[2]
        
        # Only use visible keypoints
        visible_points = []
        if nose.get("visibility", 0) > 0.5:
            visible_points.append((nose["x"], nose["y"]))
        if left_ear.get("visibility", 0) > 0.5:
            visible_points.append((left_ear["x"], left_ear["y"]))
        if right_ear.get("visibility", 0) > 0.5:
            visible_points.append((right_ear["x"], right_ear["y"]))
        
        if not visible_points:
            return default_pos
        
        # Calculate average head position
        head_x = sum(p[0] for p in visible_points) / len(visible_points)
        head_y = sum(p[1] for p in visible_points) / len(visible_points)
        
        # Convert normalized coordinates to pixel coordinates
        head_x_px = int(head_x * frame_width)
        head_y_px = int(head_y * frame_height)
        
        # Position bubble above head
        bubble_x = head_x_px - 100  # Offset left from head
        bubble_y = head_y_px - 100  # Offset above head
        
        return (bubble_x, bubble_y, head_x_px, head_y_px)
    
    def _calculate_bubble_tail(self, 
                            bubble_x: int, 
                            bubble_y: int, 
                            bubble_width: int, 
                            bubble_height: int, 
                            head_x: int, 
                            head_y: int) -> List[Tuple[int, int]]:
        """
        Calculate points for the speech bubble tail.
        
        Args:
            bubble_x, bubble_y: Top-left coordinates of bubble
            bubble_width, bubble_height: Dimensions of bubble
            head_x, head_y: Coordinates of animal's head
            
        Returns:
            List of points forming the tail polygon
        """
        # Find the closest point on the bubble rectangle to the head
        bubble_center_x = bubble_x + bubble_width // 2
        bubble_center_y = bubble_y + bubble_height // 2
        
        # Determine which side of the bubble to attach the tail
        dx = head_x - bubble_center_x
        dy = head_y - bubble_center_y
        
        if abs(dx) > abs(dy):
            # Tail attaches to left or right side
            if dx > 0:
                # Right side
                attach_x = bubble_x + bubble_width
                attach_y = bubble_y + bubble_height // 2
            else:
                # Left side
                attach_x = bubble_x
                attach_y = bubble_y + bubble_height // 2
        else:
            # Tail attaches to top or bottom side
            if dy > 0:
                # Bottom side
                attach_x = bubble_x + bubble_width // 2
                attach_y = bubble_y + bubble_height
            else:
                # Top side
                attach_x = bubble_x + bubble_width // 2
                attach_y = bubble_y
        
        # Create tail points
        tail_width = 15
        if abs(dx) > abs(dy):
            # Horizontal tail
            tail_points = [
                (attach_x, attach_y - tail_width),
                (attach_x, attach_y + tail_width),
                (head_x, head_y)
            ]
        else:
            # Vertical tail
            tail_points = [
                (attach_x - tail_width, attach_y),
                (attach_x + tail_width, attach_y),
                (head_x, head_y)
            ]
        
        return tail_points
    
    def _get_output_path(self, input_path: str) -> str:
        """
        Generate output file path based on input path.
        
        Args:
            input_path: Path to input video file
            
        Returns:
            Path to output video file
        """
        input_file = Path(input_path)
        filename = input_file.stem + "_processed" + input_file.suffix
        return str(Path(self.output_dir) / filename)

    def _ensure_web_compatible(self, video_path: str) -> str:
        """
        Ensure the video is in a web-compatible format.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Path to web-compatible video
        """
        # If we need to convert, we would do it here
        # For now, assuming the MP4 created with H.264 is compatible
        
        # Check if ffmpeg is available for conversion if needed
        try:
            import subprocess
            output_path = video_path.replace('.mp4', '_web.mp4')
            
            # Convert using ffmpeg to ensure browser compatibility
            cmd = [
                'ffmpeg', '-y', '-i', video_path, 
                '-c:v', 'libx264', '-preset', 'fast', 
                '-profile:v', 'baseline', '-level', '3.0',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                output_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"Converted video to web-compatible format: {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"Failed to convert video to web-compatible format: {e}")
            return video_path

async def process_video_file(
    video_path: str,
    animal_type: str = "auto",
    personality: str = "auto",
    display_mode: str = "speech_bubble",
    output_dir: Optional[str] = None,
    forced_state: Optional[str] = None
) -> str:
    """
    Convenience function to process a video file.
    
    Args:
        video_path: Path to input video file
        animal_type: Type of animal ("dog", "cat", or "auto")
        personality: Personality trait for dialogue generation
        display_mode: Display mode for text ("caption", "speech_bubble", or "both")
        output_dir: Directory to save processed videos
        forced_state: Override detected behavior state (e.g., "angry", "playful") - useful for testing
        
    Returns:
        Path to processed output video
    """
    processor = VideoProcessor(output_dir=output_dir)
    return await processor.process_video(
        video_path=video_path,
        animal_type=animal_type,
        personality=personality,
        display_mode=display_mode,
        forced_state=forced_state
    ) 