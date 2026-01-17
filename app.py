import streamlit as st
import whisper
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
import librosa
import numpy as np
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import gc
import platform
import textwrap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ASPECT_RATIO = 9 / 16
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
DEFAULT_CONTEXT_SECONDS = 5
MAX_CAPTION_LENGTH = 80  # Shorter for better readability
SUPPORTED_VIDEO_FORMATS = ["mp4", "mov", "avi", "mkv"]
CAPTION_MAX_CHARS_PER_LINE = 40  # Professional caption width

# Enhanced keyword list
KEYWORDS = [
    "important", "crucial", "critical", "essential", "vital",
    "success", "achievement", "accomplish", "victory", "win",
    "change", "transform", "revolution", "breakthrough",
    "remember", "learn", "understand", "realize", "discover",
    "powerful", "amazing", "incredible", "remarkable", "extraordinary",
    "key", "secret", "tip", "advice", "strategy", "best", "perfect"
]


class VideoProcessor:
    """Handles video processing with MoviePy 2.x API."""
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.cleanup_files: List[str] = []
        self.font_path = self._get_font_path()
    
    def _get_font_path(self) -> str:
        """Get the correct font path for the current OS."""
        if platform.system() == "Windows":
            possible_fonts = [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\arial.ttf",
                r"C:\Windows\Fonts\verdanab.ttf",
                r"C:\Windows\Fonts\verdana.ttf",
                r"C:\Windows\Fonts\calibrib.ttf",
                r"C:\Windows\Fonts\calibri.ttf",
            ]
        elif platform.system() == "Darwin":
            possible_fonts = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial Bold.ttf",
                "/Library/Fonts/Arial.ttf",
            ]
        else:
            possible_fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        
        for font in possible_fonts:
            if os.path.exists(font):
                logger.info(f"Using font: {font}")
                return font
        
        logger.warning(f"No font found, using default: {possible_fonts[0]}")
        return possible_fonts[0]
    
    def _format_caption_professional(self, text: str) -> str:
        """Format caption like professional movie subtitles."""
        # Clean up the text
        text = text.strip()
        
        # Remove incomplete sentences at the end
        if text and not text[-1] in '.!?':
            # Find last complete sentence
            last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_period > 20:  # Only cut if we have enough text
                text = text[:last_period + 1]
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Wrap text to max line width (like movie subtitles)
        lines = textwrap.wrap(text, width=CAPTION_MAX_CHARS_PER_LINE)
        
        # Limit to 2 lines max (standard for subtitles)
        if len(lines) > 2:
            # Take first 2 lines
            lines = lines[:2]
            # Remove incomplete last word and add ellipsis
            last_line = lines[1]
            words = last_line.split()
            if len(words) > 1:
                lines[1] = ' '.join(words[:-1]) + '...'
        
        return "\n".join(lines)
    
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
        audio_path = str(self.temp_dir / "audio.wav")
        try:
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path, logger=None)
            clip.close()
            self.cleanup_files.append(audio_path)
            return audio_path
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, model_name: str = "tiny") -> Dict:
        """Transcribe audio using Whisper (optimized for speed)."""
        try:
            model = whisper.load_model(model_name)
            # Use faster settings
            result = model.transcribe(
                audio_path, 
                fp16=False,
                language="en",  # Specify language for speed
                verbose=False
            )
            return result
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def analyze_audio_energy(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Analyze audio energy levels using RMS (optimized)."""
        try:
            # Use lower sample rate for faster processing
            y, sr = librosa.load(audio_path, sr=16000)
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            
            if np.max(rms) > np.min(rms):
                rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
            else:
                rms_normalized = rms
            return rms_normalized, sr
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            raise
    
    def score_segment(self, segment: Dict, rms: np.ndarray, sr: int, 
                     hop_length: int = 512) -> float:
        """Score a segment based on audio energy and keyword presence."""
        mid_time = (segment["start"] + segment["end"]) / 2
        frame_idx = int(mid_time * sr / hop_length)
        
        if frame_idx < len(rms):
            audio_score = rms[frame_idx]
        else:
            audio_score = 0
        
        text = segment["text"].lower()
        keyword_score = sum(1 for kw in KEYWORDS if kw in text)
        
        # Weighted: audio 40%, keywords 60%
        total_score = (audio_score * 0.4) + (keyword_score * 0.6)
        return total_score
    
    def find_best_segments(self, segments: List[Dict], rms: np.ndarray, 
                          sr: int, top_n: int = 5) -> List[Tuple[Dict, float]]:
        """Find the top N best segments based on scoring."""
        if not segments:
            raise ValueError("No segments found in transcription")
        
        # Score all segments
        scored_segments = []
        for seg in segments:
            score = self.score_segment(seg, rms, sr)
            scored_segments.append((seg, score))
        
        # Sort by score (highest first) and return top N
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        return scored_segments[:top_n]
    
    def create_short_clip(self, video_path: str, start: float, end: float,
                         caption: str, output_path: str) -> None:
        """Create a vertical short clip with professional movie-style captions."""
        clip = None
        short = None
        short_cropped = None
        short_resized = None
        txt = None
        bg_box = None
        final = None
        
        try:
            # Load video
            clip = VideoFileClip(video_path)
            
            # Cut clip
            short = clip.subclipped(start, end)
            
            # Get dimensions
            w, h = short.size
            target_w = int(h * ASPECT_RATIO)
            
            # Center crop
            x_center = w / 2
            x1 = max(0, int(x_center - target_w / 2))
            y1 = 0
            width = min(target_w, w - x1)
            height = h
            
            # Crop and resize
            short_cropped = short.cropped(x1=x1, y1=y1, width=width, height=height)
            short_resized = short_cropped.resized((OUTPUT_WIDTH, OUTPUT_HEIGHT))
            
            # Format caption professionally
            formatted_caption = self._format_caption_professional(caption)
            
            # Create Netflix-style text - SMALLER font size
            txt = TextClip(
                text=formatted_caption,
                font=self.font_path,
                font_size=38,  # Much smaller - reduced from 48
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(OUTPUT_WIDTH - 160, None)  # More margin for smaller text
            )
            
            # Create semi-transparent black background box that fits text perfectly
            bg_width = min(txt.w + 50, OUTPUT_WIDTH - 60)  # Compact background
            bg_height = txt.h + 24  # Tighter padding
            
            bg_box = ColorClip(
                size=(bg_width, bg_height),
                color=(0, 0, 0)
            ).with_opacity(0.8)
            
            # Position HIGHER from bottom - 200px to ensure it's fully visible
            bottom_margin = 200  # Moved up significantly from 120
            
            # Set durations
            txt = txt.with_duration(short_resized.duration)
            bg_box = bg_box.with_duration(short_resized.duration)
            
            # Position background box centered and higher
            bg_y = OUTPUT_HEIGHT - bottom_margin - bg_height
            bg_box = bg_box.with_position(("center", bg_y))
            
            # Position text centered on background
            txt_y = bg_y + 12  # 12px padding from top of bg
            txt = txt.with_position(("center", txt_y))
            
            # Composite: video + background box + text
            final = CompositeVideoClip([short_resized, bg_box, txt])
            
            # Write with faster settings
            final.write_videofile(
                output_path,
                fps=30,
                codec="libx264",
                audio_codec="aac",
                preset="ultrafast",  # Faster encoding
                threads=4,  # Use multiple threads
                logger=None
            )
            
        except Exception as e:
            logger.error(f"Clip creation failed: {e}")
            raise
        finally:
            for obj in [txt, bg_box, final, short_resized, short_cropped, short, clip]:
                if obj is not None:
                    try:
                        obj.close()
                    except:
                        pass
            gc.collect()
    
    def cleanup(self):
        """Clean up temporary files."""
        for file_path in self.cleanup_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PulsePoint AI",
        page_icon="🎯",
        layout="centered"
    )
    
    st.title("🎯 PulsePoint AI — Best Moments from Long Videos")
    st.markdown("""
    Upload a long video and let AI automatically identify and extract 
    the most engaging moment as a vertical short video with **professional movie-style captions**!
    """)
    
    # File uploader
    video_file = st.file_uploader(
        "Upload your video",
        type=SUPPORTED_VIDEO_FORMATS,
        help="Supported formats: MP4, MOV, AVI, MKV"
    )
    
    if video_file:
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        input_video = temp_path / "input.mp4"
        output_video = temp_path / "final_short.mp4"
        
        try:
            with open(input_video, "wb") as f:
                f.write(video_file.read())
            
            st.video(str(input_video))
            
            # Processing controls
            col1, col2 = st.columns(2)
            with col1:
                num_clips = st.selectbox(
                    "Number of Clips",
                    [1, 3, 5],
                    index=2,  # Default to 5
                    help="Generate multiple best moments"
                )
            
            with col2:
                model_size = st.selectbox(
                    "AI Model",
                    ["tiny", "base", "small"],
                    index=0,
                    help="Tiny = Fastest"
                )
            
            context_seconds = st.slider(
                "Context (seconds before/after)",
                min_value=0,
                max_value=10,
                value=DEFAULT_CONTEXT_SECONDS,
                help="Add context around moments"
            )
            
            if st.button("🔥 Generate Best Shorts", type="primary"):
                processor = VideoProcessor(temp_path)
                progress_bar = st.progress(0)
                
                try:
                    # Step 1: Extract Audio
                    with st.spinner("📤 Extracting audio..."):
                        audio_path = processor.extract_audio(str(input_video))
                    progress_bar.progress(15)
                    
                    # Step 2: Transcribe
                    with st.spinner("🎙️ Transcribing (using fast AI model)..."):
                        result = processor.transcribe_audio(audio_path, model_size)
                        segments = result["segments"]
                    progress_bar.progress(40)
                    
                    # Step 3: Analyze Audio
                    with st.spinner("📊 Analyzing audio energy..."):
                        rms, sr = processor.analyze_audio_energy(audio_path)
                    progress_bar.progress(55)
                    
                    # Step 4: Find Best Segments
                    with st.spinner(f"🔍 Finding top {num_clips} moments..."):
                        best_segments = processor.find_best_segments(segments, rms, sr, num_clips)
                    progress_bar.progress(65)
                    
                    st.success(f"✅ Found {len(best_segments)} best moments!")
                    
                    # Show all found moments
                    with st.expander("📝 View All Detected Moments"):
                        for idx, (seg, score) in enumerate(best_segments, 1):
                            st.write(f"**#{idx} - Score: {score:.2f}**")
                            st.write(f"Text: {seg['text'][:80]}...")
                            st.write(f"Time: {seg['start']:.1f}s - {seg['end']:.1f}s")
                            st.divider()
                    
                    # Get video duration
                    clip = VideoFileClip(str(input_video))
                    video_duration = clip.duration
                    clip.close()
                    
                    # Step 5: Create All Shorts
                    generated_videos = []
                    
                    for idx, (best_seg, score) in enumerate(best_segments, 1):
                        with st.spinner(f"✂️ Creating short #{idx}/{num_clips}..."):
                            output_file = temp_path / f"short_{idx}.mp4"
                            
                            start = max(best_seg["start"] - context_seconds, 0)
                            end = min(best_seg["end"] + context_seconds, video_duration)
                            
                            caption = best_seg["text"][:MAX_CAPTION_LENGTH]
                            
                            processor.create_short_clip(
                                str(input_video),
                                start,
                                end,
                                caption,
                                str(output_file)
                            )
                            
                            generated_videos.append((output_file, score, best_seg))
                            
                            # Update progress
                            progress = 65 + int((idx / num_clips) * 35)
                            progress_bar.progress(progress)
                    
                    progress_bar.progress(100)
                    st.success(f"✅ Generated {num_clips} professional short videos!")
                    
                    # Display all videos
                    st.markdown("---")
                    st.subheader("🎬 Generated Short Videos")
                    
                    for idx, (video_path, score, seg) in enumerate(generated_videos, 1):
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.markdown(f"### Short #{idx} (Score: {score:.2f})")
                            st.video(str(video_path))
                        
                        with col_b:
                            st.markdown(f"**Time:** {seg['start']:.1f}s - {seg['end']:.1f}s")
                            st.markdown(f"**Caption:** {seg['text'][:60]}...")
                            
                            # Download button for each
                            with open(video_path, "rb") as f:
                                st.download_button(
                                    label=f"⬇️ Download #{idx}",
                                    data=f,
                                    file_name=f"pulsepoint_short_{idx}.mp4",
                                    mime="video/mp4",
                                    key=f"download_{idx}"
                                )
                        
                        st.markdown("---")
                    
                    processor.cleanup()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.exception("Processing failed")
                    import traceback
                    with st.expander("🔍 Debug Info"):
                        st.code(traceback.format_exc())
        
        finally:
            try:
                import time
                time.sleep(1)
                gc.collect()
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")


if __name__ == "__main__":
    main()