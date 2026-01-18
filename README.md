# 🎯 PulsePoint AI
> Hackathon Submission — ByteSize Sage AI Hackathon (Unstop)


> Transform long-form videos into viral-ready short clips using multimodal AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

PulsePoint AI is an intelligent video editor that automatically identifies and extracts the most engaging moments from long videos, creating vertical short-form content perfect for TikTok, Instagram Reels, and YouTube Shorts. Using multimodal AI analysis, it combines speech recognition, audio energy detection, and keyword scoring to find the best clips.

## ✨ Features

- 🎙️ **AI Speech Recognition** - Transcribes video content with precise timestamps using OpenAI Whisper
- 📊 **Audio Energy Analysis** - Detects exciting moments using Librosa RMS energy computation
- 🔍 **Smart Keyword Detection** - Identifies impactful phrases like "important," "breakthrough," "success"
- 🎬 **Multimodal Scoring** - Combines audio and text analysis for intelligent moment selection
- 📱 **Vertical Video Export** - Automatically converts to 9:16 aspect ratio (1080x1920)
- 💬 **Professional Captions** - Adds Netflix-style subtitles with semi-transparent backgrounds
- 🚀 **Batch Processing** - Generate up to 5 short clips from a single video
- ⚡ **Fast Processing** - Optimized pipeline with configurable AI model sizes

## 🧠 Multimodal Reasoning

PulsePoint AI uses a sophisticated multimodal scoring system to identify the best moments:

### 1. Audio Analysis (40% weight)
- Computes RMS (Root Mean Square) energy across the audio timeline
- Identifies peaks indicating excitement, emphasis, or emotional intensity
- Normalizes values for consistent scoring

### 2. Text Analysis (60% weight)
- Transcribes speech using Whisper AI
- Scans for 25+ high-value keywords across categories:
  - **Impact**: important, crucial, critical, essential
  - **Achievement**: success, victory, breakthrough
  - **Transformation**: change, revolution, transform
  - **Learning**: remember, discover, understand
  - **Emphasis**: powerful, incredible, remarkable

### 3. Combined Scoring
```python
score = (audio_energy * 0.4) + (keyword_count * 0.6)
```

This weighted approach ensures clips contain both energetic delivery AND meaningful content, not just loud moments or keyword spam.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **AI Transcription** | OpenAI Whisper |
| **Audio Analysis** | Librosa |
| **Video Processing** | MoviePy 2.x |
| **Video Codec** | FFmpeg |
| **Language** | Python 3.10+ |

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10 or higher
- FFmpeg installed and available in PATH
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pulsepoint-ai.git
cd pulsepoint-ai
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install streamlit openai-whisper moviepy librosa numpy soundfile torch
```

4. **Install FFmpeg** (if not already installed)
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

5. **Run the app**
```bash
streamlit run app.py
```

6. **Open browser**
Navigate to `http://localhost:8501`

## 📹 Demo Video

Watch PulsePoint AI in action:

**[🎥 View Demo Video]( https://drive.google.com/drive/folders/1WwpKATIvyRr1eQNcYr_3PpWd1z4dYljz?usp=sharing )**

*Add your demo video link from YouTube, Loom, or any video hosting platform*

## 🎮 Usage

1. Upload a video file (MP4, MOV, AVI, MKV)
2. Select number of clips to generate (1, 3, or 5)
3. Choose AI model size (tiny/base/small)
4. Adjust context duration (0-10 seconds)
5. Click "🔥 Generate Best Shorts"
6. Download your generated clips!

## 📊 Performance

- **Tiny Model**: ~2-3 minutes for 10-minute video
- **Base Model**: ~4-5 minutes for 10-minute video
- **Small Model**: ~6-8 minutes for 10-minute video

*Times vary based on hardware*

## 🎯 Use Cases

- 📚 **Education**: Extract key teaching moments from lectures
- 🎤 **Podcasts**: Create shareable clips from long episodes
- 🎬 **Content Creation**: Repurpose YouTube videos for TikTok/Reels
- 💼 **Marketing**: Generate promotional clips from webinars
- 🎮 **Gaming**: Highlight best moments from streams

## 🔮 Future Enhancements

- [ ] Face detection for better center-crop positioning
- [ ] Multiple aspect ratio support (1:1, 16:9, 4:5)
- [ ] Custom keyword lists
- [ ] Emotion detection using facial analysis
- [ ] Automatic hashtag generation
- [ ] Direct social media upload

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests



## 👨‍💻 Author

**Your Name**
- GitHub: [@nithin2113](https://github.com/nithin2113)

## 🙏 Acknowledgments

- OpenAI for Whisper speech recognition
- Zulko for MoviePy video processing
- Librosa team for audio analysis tools
- Streamlit for the amazing web framework

---

**Built with ❤️ for ByteSize Sage Ai Hackathon**

*If you found this project helpful, please give it a ⭐!*
