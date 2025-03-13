# Pet Video Dialogue: System Architecture

This document outlines the architecture of the Pet Video Dialogue system, which processes videos of pets, detects their poses, and overlays humorous AI-generated dialogue.

## System Overview

The system processes pre-recorded videos of pets (currently dogs and cats), analyzes their body language frame-by-frame, and generates contextual dialogue based on detected behaviors. The processed video is then rendered with the dialogue overlaid as speech bubbles or captions.

## System Components

### 1. Video Processing Pipeline

```
┌───────────┐    ┌───────────┐    ┌────────────┐    ┌────────────┐    ┌───────────┐
│  Upload   │ ─> │  Extract  │ ─> │    Pose    │ ─> │  Dialogue  │ ─> │  Render   │
│  Handler  │    │  Frames   │    │ Detection  │    │ Generation │    │  Output   │
└───────────┘    └───────────┘    └────────────┘    └────────────┘    └───────────┘
```

### 2. Server Backend (FastAPI)

- **Upload Endpoint**: Handles video file uploads
- **Task Management**: Tracks processing status and progress
- **Asynchronous Processing**: Processes videos without blocking
- **Result Delivery**: Serves processed videos for download

### 3. Core Processing Modules

#### VideoProcessor

Manages the overall video processing workflow:
- Reads video files frame-by-frame
- Coordinates pose detection, behavior classification, and dialogue generation
- Renders text overlays (speech bubbles or captions)
- Exports processed video files

#### PoseDetector

Uses MediaPipe to detect animal poses:
- Identifies key skeletal points in each frame
- Normalizes pose coordinates
- Detects animal type (dog/cat) if not specified

#### BehaviorClassifier

Maps detected poses to behavioral states:
- Uses rule-based classification
- Calculates confidence scores for each state
- Maintains temporal consistency in classifications

#### DialogueGenerator

Generates contextual dialogue based on behavior:
- Powered by TinyLlama-1.1B-Chat model
- Contextual dialogue generation based on:
  - Detected behavior
  - Animal type
  - Selected personality
  - Scene context
- Personality traits:
  - Enthusiastic
  - Sassy
  - Dramatic
  - Philosophical
  - Sarcastic

#### PersonalityEngine

Manages different dialogue styles:
- Adjusts dialogue style and tone
- Maintains consistent character voice
- Handles personality-specific responses

## Data Flow

1. **User uploads video** through web interface, specifying:
   - Animal type (dog, cat, auto-detect)
   - Personality trait
   - Display mode

2. **Server creates a processing task** and begins asynchronous processing

3. **Frame-by-frame processing**:
   - Extract frame from video
   - Detect animal pose with MediaPipe
   - Classify behavior based on pose
   - Generate dialogue if behavior changed or timer expired
   - Render dialogue overlay (speech bubble or caption)
   - Add watermark

4. **Output video rendered** with all processed frames

5. **User downloads** the processed video file

## Technical Implementation Details

### Video Processing

- Uses OpenCV for frame extraction and video writing
- Processes frames at adaptive intervals for efficiency
- Handles various input formats (MP4, AVI, MOV)

### Pose Detection

- MediaPipe animal pose model detects 17 keypoints
- Keypoints include ears, nose, shoulders, hips, limbs, and tail
- Normalizes coordinates relative to frame dimensions

### Behavior Classification

- Rule-based classifier using threshold values for features
- Features derived from relative joint positions and angles
- States include: excited, anxious, curious, playful, relaxed, alert

### Dialogue Generation

- Primary: TinyLlama-1.1B-Chat model
- Fallback: Template-based generation using predefined examples
- Ethical guardrails prevent anthropomorphizing distress signals

### Speech Bubble Rendering

- Dynamic positioning based on head location
- Adjustable opacity and styling
- Tail points toward the animal's head location

## User Interface

- Bootstrap-based responsive web interface
- Upload form with settings configuration
- Real-time progress tracking
- Video playback with download option

## Performance Considerations

- Target processing time: Maximum 2x video duration
- Smart frame skipping for efficiency
- Dialogue persistence ensures smooth transitions
- Progress tracking via task status API

## Future Enhancements

- Multi-animal support
- Advanced video preprocessing for quality improvement
- Expanded animal type support
- Custom personality creator
- Enhanced visual effects
- Mobile app version

## Ethical Considerations

- Watermarking for "entertainment only"
- Guardrails against anthropomorphizing distress
- Privacy-focused design (no storage of user videos beyond processing)
- Clear communication about AI-generated content 