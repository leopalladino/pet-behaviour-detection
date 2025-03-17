# Pet Behavior Detection & Dialogue Generation  

Add humorous dialogue to pet videos using pose detection and AI.  

## Key Features  
- 🐶 **Animal Support**: Dogs and cats (auto-detection available)  
- 📹 **Video Processing**: Upload pre-recorded pet videos  
- 🦴 **Pose Detection**: MediaPipe body language analysis  
- 🎭 **Personality Modes**: 5 unique dialogue styles  
- 💬 **Display Options**: Speech bubbles or captions  

## Example Results  

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Happy Dog</b></td>
      <td align="center"><b>Angry Dog</b></td>
    </tr>
    <tr>
      <td align="center"><img src="sample_output/sample_happy_output.gif" width="400"></td>
      <td align="center"><img src="sample_output/sample_angry_output.gif" width="400"></td>
    </tr>
  </table>
</div>

## Installation  

```bash
git clone https://github.com/leopalladino/pet-behaviour-detection.git
cd pet-behaviour-detection
pip install -r requirements.txt
```

## Usage

1. **Start the Server**
   ```bash
   python app.py
   ```

2. **Access the Web Interface**
   - Open your browser to `http://localhost:8000`

3. **Process a Video**
   - Upload your pet video (MP4, AVI, MOV)
   - Choose settings:
     - Animal Type: Auto-detect, Dog, or Cat
     - Personality: Auto-adapt, Enthusiastic, Sassy, Dramatic, Philosophical, or Sarcastic
     - Display: Speech Bubble, Caption, or Both
   - Click "Process Video"
   - Wait for processing to complete
   - Download or preview the result

### Tips
- Use well-lit videos with your pet clearly visible
- Keep videos 15-60 seconds long
- Make sure face, tail, and body are visible
- Record in 720p or higher resolution

Note: For entertainment purposes only. Generated dialogue does not reflect actual pet thoughts.
