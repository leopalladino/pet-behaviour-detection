#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark script for Pet Video Dialogue.
Evaluates the performance of the video processing system across various metrics.
"""

import os
import time
import argparse
import asyncio
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from pet_dialogue.utils import setup_logging
from pet_dialogue.pose_detector import PoseDetector
from pet_dialogue.behavior_classifier import BehaviorClassifier
from video_processor import VideoProcessor

# Initialize logging
setup_logging()

class BenchmarkRunner:
    """
    Runs benchmark tests for the Pet Video Dialogue system.
    """
    
    def __init__(self, video_dir: str, output_dir: Optional[str] = None):
        """
        Initialize the benchmark runner.
        
        Args:
            video_dir: Directory containing test videos
            output_dir: Directory to save benchmark results
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components for individual benchmarks
        self.pose_detector = PoseDetector()
        self.behavior_classifier = BehaviorClassifier()
        
        # Initialize VideoProcessor for end-to-end tests
        self.video_processor = VideoProcessor(output_dir=str(self.output_dir))
        
        # Results dictionary
        self.results = {
            "pose_detection": {},
            "behavior_classification": {},
            "end_to_end": {}
        }
    
    async def run_benchmarks(self):
        """Run all benchmarks and collect results."""
        logger.info("Starting benchmark tests")
        
        # Collect test videos
        test_videos = self._collect_test_videos()
        if not test_videos:
            logger.error(f"No test videos found in {self.video_dir}")
            return
        
        logger.info(f"Found {len(test_videos)} test videos")
        
        # Run component benchmarks
        self._benchmark_pose_detection(test_videos)
        self._benchmark_behavior_classification(test_videos)
        
        # Run end-to-end benchmarks
        await self._benchmark_end_to_end(test_videos)
        
        # Save and report results
        self._save_results()
        self._report_results()
    
    def _collect_test_videos(self) -> List[Path]:
        """
        Collect test videos from the video directory.
        
        Returns:
            List of paths to test videos
        """
        if not self.video_dir.exists():
            logger.error(f"Video directory does not exist: {self.video_dir}")
            return []
        
        return list(self.video_dir.glob("**/*.[mM][pP]4"))
    
    def _benchmark_pose_detection(self, test_videos: List[Path]):
        """
        Benchmark pose detection speed and accuracy.
        
        Args:
            test_videos: List of paths to test videos
        """
        logger.info("Benchmarking pose detection")
        
        results = {}
        
        for video_path in test_videos:
            video_name = video_path.name
            logger.info(f"Processing {video_name}")
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                continue
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            # Process frames
            detection_times = []
            detection_rates = []
            frame_count = 0
            detection_count = 0
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Skip frames (process every 5th frame)
                    if frame_count % 5 != 0:
                        continue
                    
                    # Encode frame
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Time pose detection
                    start_time = time.time()
                    pose_results = self.pose_detector.process_frame(frame_bytes)
                    end_time = time.time()
                    
                    detection_time = end_time - start_time
                    detection_times.append(detection_time)
                    
                    # Count successful detections
                    if pose_results and pose_results.get("keypoints"):
                        detection_count += 1
                
                # Calculate metrics
                processed_frames = frame_count // 5
                detection_rate = detection_count / processed_frames if processed_frames > 0 else 0
                avg_detection_time = statistics.mean(detection_times) if detection_times else 0
                detection_rates.append(detection_rate)
                
                # Store results
                results[video_name] = {
                    "avg_detection_time": avg_detection_time,
                    "detection_rate": detection_rate,
                    "total_frames": total_frames,
                    "duration": duration,
                    "detection_count": detection_count
                }
                
                logger.info(f"Detection rate: {detection_rate:.2f}, Avg time: {avg_detection_time:.4f}s")
            
            finally:
                cap.release()
        
        # Calculate overall metrics
        if detection_times:
            results["overall"] = {
                "avg_detection_time": statistics.mean(detection_times),
                "median_detection_time": statistics.median(detection_times),
                "min_detection_time": min(detection_times),
                "max_detection_time": max(detection_times),
                "avg_detection_rate": statistics.mean(detection_rates) if detection_rates else 0
            }
        
        self.results["pose_detection"] = results
    
    def _benchmark_behavior_classification(self, test_videos: List[Path]):
        """
        Benchmark behavior classification speed and consistency.
        
        Args:
            test_videos: List of paths to test videos
        """
        logger.info("Benchmarking behavior classification")
        
        results = {}
        
        for video_path in test_videos:
            video_name = video_path.name
            logger.info(f"Processing {video_name}")
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                continue
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            # Process frames
            classification_times = []
            behaviors = []
            confidences = []
            frame_count = 0
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Skip frames (process every 5th frame)
                    if frame_count % 5 != 0:
                        continue
                    
                    # Encode frame
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Detect pose
                    pose_results = self.pose_detector.process_frame(frame_bytes)
                    
                    if pose_results and pose_results.get("keypoints"):
                        # Time behavior classification
                        start_time = time.time()
                        behavior = self.behavior_classifier.classify(pose_results)
                        end_time = time.time()
                        
                        classification_time = end_time - start_time
                        classification_times.append(classification_time)
                        
                        # Track behavior states and confidences
                        if behavior:
                            behaviors.append(behavior.get("state", "unknown"))
                            confidences.append(behavior.get("confidence", 0.0))
                
                # Calculate metrics
                avg_classification_time = statistics.mean(classification_times) if classification_times else 0
                avg_confidence = statistics.mean(confidences) if confidences else 0
                
                # Count behavior states
                behavior_counts = {}
                for behavior in behaviors:
                    behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                
                # Calculate behavior stability (percentage of most common behavior)
                most_common_behavior = max(behavior_counts.items(), key=lambda x: x[1]) if behavior_counts else ("unknown", 0)
                behavior_stability = most_common_behavior[1] / len(behaviors) if behaviors else 0
                
                # Store results
                results[video_name] = {
                    "avg_classification_time": avg_classification_time,
                    "avg_confidence": avg_confidence,
                    "behavior_stability": behavior_stability,
                    "most_common_behavior": most_common_behavior[0],
                    "behavior_counts": behavior_counts,
                    "total_frames": total_frames,
                    "duration": duration
                }
                
                logger.info(f"Avg classification time: {avg_classification_time:.4f}s, Avg confidence: {avg_confidence:.2f}")
            
            finally:
                cap.release()
        
        # Calculate overall metrics
        if classification_times:
            results["overall"] = {
                "avg_classification_time": statistics.mean(classification_times),
                "median_classification_time": statistics.median(classification_times),
                "min_classification_time": min(classification_times),
                "max_classification_time": max(classification_times),
                "avg_confidence": statistics.mean(confidences) if confidences else 0
            }
        
        self.results["behavior_classification"] = results
    
    async def _benchmark_end_to_end(self, test_videos: List[Path]):
        """
        Benchmark end-to-end processing time and resource usage.
        
        Args:
            test_videos: List of paths to test videos
        """
        logger.info("Benchmarking end-to-end processing")
        
        results = {}
        
        for video_path in test_videos:
            video_name = video_path.name
            logger.info(f"Processing {video_name}")
            
            # Get video duration
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()
            
            # Test different configurations
            configs = [
                ("auto", "auto", "speech_bubble"),
                ("dog", "enthusiastic", "caption"),
                ("cat", "sarcastic", "both")
            ]
            
            config_results = {}
            
            for animal_type, personality, display_mode in configs:
                config_name = f"{animal_type}_{personality}_{display_mode}"
                logger.info(f"Testing configuration: {config_name}")
                
                try:
                    # Time end-to-end processing
                    start_time = time.time()
                    output_path = await self.video_processor.process_video(
                        video_path=str(video_path),
                        animal_type=animal_type,
                        personality=personality,
                        display_mode=display_mode
                    )
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    
                    # Get output video size
                    output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                    
                    # Calculate metrics
                    processing_ratio = processing_time / duration if duration > 0 else 0
                    
                    # Store results
                    config_results[config_name] = {
                        "processing_time": processing_time,
                        "processing_ratio": processing_ratio,
                        "output_size": output_size,
                        "input_duration": duration,
                        "output_path": output_path
                    }
                    
                    logger.info(f"Processing time: {processing_time:.2f}s ({processing_ratio:.2f}x real-time)")
                
                except Exception as e:
                    logger.error(f"Error processing {video_name} with config {config_name}: {e}")
                    config_results[config_name] = {"error": str(e)}
            
            results[video_name] = config_results
        
        self.results["end_to_end"] = results
    
    def _save_results(self):
        """Save benchmark results to a file."""
        import json
        
        # Convert Path objects to strings
        def path_to_str(obj):
            if isinstance(obj, Path):
                return str(obj)
            return obj
        
        result_path = self.output_dir / "benchmark_results.json"
        with open(result_path, "w") as f:
            json.dump(self.results, f, default=path_to_str, indent=2)
        
        logger.info(f"Benchmark results saved to {result_path}")
    
    def _report_results(self):
        """Print a summary of benchmark results."""
        logger.info("Benchmark Summary:")
        
        # Pose detection summary
        if "overall" in self.results["pose_detection"]:
            pd = self.results["pose_detection"]["overall"]
            logger.info("Pose Detection:")
            logger.info(f"  Avg Time: {pd['avg_detection_time']:.4f}s")
            logger.info(f"  Detection Rate: {pd['avg_detection_rate']:.2%}")
        
        # Behavior classification summary
        if "overall" in self.results["behavior_classification"]:
            bc = self.results["behavior_classification"]["overall"]
            logger.info("Behavior Classification:")
            logger.info(f"  Avg Time: {bc['avg_classification_time']:.4f}s")
            logger.info(f"  Avg Confidence: {bc['avg_confidence']:.2f}")
        
        # End-to-end summary
        if self.results["end_to_end"]:
            logger.info("End-to-End Processing:")
            
            # Collect all processing ratios
            all_ratios = []
            for video_results in self.results["end_to_end"].values():
                for config_results in video_results.values():
                    if "processing_ratio" in config_results:
                        all_ratios.append(config_results["processing_ratio"])
            
            if all_ratios:
                avg_ratio = statistics.mean(all_ratios)
                logger.info(f"  Avg Processing Ratio: {avg_ratio:.2f}x real-time")


async def main():
    """Run benchmark tests."""
    parser = argparse.ArgumentParser(description="Benchmark the Pet Video Dialogue system")
    parser.add_argument("--video_dir", required=True, help="Directory containing test videos")
    parser.add_argument("--output_dir", default="benchmark_results", help="Directory to save benchmark results")
    
    args = parser.parse_args()
    
    benchmark = BenchmarkRunner(args.video_dir, args.output_dir)
    await benchmark.run_benchmarks()


if __name__ == "__main__":
    asyncio.run(main()) 