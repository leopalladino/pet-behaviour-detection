#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main application file for Pet Video Dialogue.
Provides a web interface for processing videos of pets and generating overlaid dialogue.
"""

import os
import time
import logging
import asyncio
import tempfile
import shutil
from typing import List, Optional, Dict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pet_dialogue.utils import setup_logging
from video_processor import VideoProcessor, process_video_file

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Pet Video Dialogue",
    description="Process videos of pets and generate overlaid dialogue based on their behavior.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temporary directories
TEMP_DIR = Path(tempfile.gettempdir()) / "pet_dialogue"
UPLOAD_DIR = TEMP_DIR / "uploads"
OUTPUT_DIR = TEMP_DIR / "outputs"

for directory in [UPLOAD_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store active processing tasks
processing_tasks = {}


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page."""
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    animal_type: str = Form("auto"),
    personality: str = Form("auto"),
    display_mode: str = Form("speech_bubble"),
    forced_state: Optional[str] = Form(None)
):
    """
    Upload a video file for processing.
    
    Args:
        video: The video file to upload
        animal_type: Type of animal ("dog", "cat", or "auto")
        personality: Personality trait for dialogue generation
        display_mode: Display mode for text ("caption", "speech_bubble", or "both")
        forced_state: Override detected behavior state (e.g., "angry", "playful") - useful for testing
    
    Returns:
        Task ID for the processing job
    """
    # Validate video file
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    filename = video.filename.lower()
    if not any(filename.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Generate unique file name
    timestamp = int(time.time())
    unique_filename = f"{timestamp}_{video.filename}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save uploaded file
    with open(file_path, "wb") as f:
        content = await video.read()
        f.write(content)
    
    logger.info(f"Uploaded video: {file_path}")
    
    # Generate a task ID
    task_id = f"task_{timestamp}"
    
    # Start processing in the background
    task = asyncio.create_task(
        process_video_task(
            task_id=task_id,
            file_path=str(file_path),
            animal_type=animal_type,
            personality=personality,
            display_mode=display_mode,
            forced_state=forced_state
        )
    )
    
    # Store task reference
    processing_tasks[task_id] = {
        "task": task,
        "status": "processing",
        "start_time": time.time(),
        "file_path": str(file_path),
        "animal_type": animal_type,
        "personality": personality,
        "display_mode": display_mode,
        "forced_state": forced_state
    }
    
    return {"task_id": task_id, "status": "processing"}


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a processing task.
    
    Args:
        task_id: Task ID to check
    
    Returns:
        Task status information
    """
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = processing_tasks[task_id]
    
    # Calculate progress if task is still processing
    if task_info["status"] == "processing":
        # Calculate elapsed time
        elapsed_time = time.time() - task_info["start_time"]
        
        # Estimate progress (very rough estimate)
        # Assuming processing takes ~2x video duration
        progress = min(95, int(elapsed_time / 20 * 100))  # Cap at 95%
        
        return {
            "status": "processing",
            "progress": progress,
            "elapsed_time": int(elapsed_time)
        }
    
    # If task is complete, return result info
    elif task_info["status"] == "complete":
        return {
            "status": "complete",
            "result": task_info.get("result", {}),
            "download_url": f"/download/{task_id}"
        }
    
    # If task failed, return error info
    elif task_info["status"] == "failed":
        return {
            "status": "failed",
            "error": task_info.get("error", "Unknown error")
        }
    
    # Default fallback
    return {"status": task_info["status"]}


@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """
    Download a processed video.
    
    Args:
        task_id: Task ID of the processed video
    
    Returns:
        Processed video file
    """
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = processing_tasks[task_id]
    
    if task_info["status"] != "complete":
        raise HTTPException(status_code=400, detail="Processing not complete")
    
    output_path = task_info.get("result", {}).get("output_path")
    
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    logger.info(f"Serving video file: {output_path}")
    
    # Create response with explicit content type and cache control
    return FileResponse(
        path=output_path,
        filename=os.path.basename(output_path),
        media_type="video/mp4",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a processing task and its associated files.
    
    Args:
        task_id: Task ID to delete
    
    Returns:
        Success message
    """
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = processing_tasks[task_id]
    
    # Cancel task if still running
    if task_info["status"] == "processing":
        task_info["task"].cancel()
    
    # Delete input file if it exists
    input_path = task_info.get("file_path")
    if input_path and os.path.exists(input_path):
        os.remove(input_path)
    
    # Delete output file if it exists
    output_path = task_info.get("result", {}).get("output_path")
    if output_path and os.path.exists(output_path):
        os.remove(output_path)
    
    # Remove task from tracking
    del processing_tasks[task_id]
    
    return {"status": "success", "message": "Task deleted"}


async def process_video_task(
    task_id: str,
    file_path: str,
    animal_type: str,
    personality: str,
    display_mode: str,
    forced_state: Optional[str] = None
):
    """
    Background task to process a video file.
    
    Args:
        task_id: Task ID
        file_path: Path to the uploaded video file
        animal_type: Type of animal ("dog", "cat", or "auto")
        personality: Personality trait for dialogue generation
        display_mode: Display mode for text ("caption", "speech_bubble", or "both")
        forced_state: Override detected behavior state (e.g., "angry", "playful") - useful for testing
    """
    try:
        logger.info(f"Processing task {task_id}: {file_path}")
        
        # Process the video
        output_path = await process_video_file(
            video_path=file_path,
            animal_type=animal_type,
            personality=personality,
            display_mode=display_mode,
            output_dir=str(OUTPUT_DIR),
            forced_state=forced_state
        )
        
        # Update task status
        processing_tasks[task_id]["status"] = "complete"
        processing_tasks[task_id]["result"] = {
            "output_path": output_path,
            "animal_type": animal_type,
            "personality": personality,
            "display_mode": display_mode,
            "forced_state": forced_state
        }
        
        logger.info(f"Task {task_id} completed: {output_path}")
        
    except asyncio.CancelledError:
        logger.info(f"Task {task_id} cancelled")
        processing_tasks[task_id]["status"] = "cancelled"
        
    except Exception as e:
        logger.exception(f"Task {task_id} failed: {str(e)}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up temporary files on shutdown."""
    logger.info("Shutting down, cleaning up temporary files")
    
    # Cancel all running tasks
    for task_id, task_info in processing_tasks.items():
        if task_info["status"] == "processing":
            task_info["task"].cancel()
    
    # Clean up temp directories
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        if directory.exists():
            try:
                shutil.rmtree(directory)
                logger.info(f"Removed directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to remove directory {directory}: {e}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 