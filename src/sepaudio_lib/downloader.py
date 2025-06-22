import uuid
from pathlib import Path
import yt_dlp
from .utils import sanitize_filename

try:
    from moviepy.editor import AudioFileClip
except ImportError:
    AudioFileClip = None

def download_audio_from_url(url: str, output_dir: Path) -> Path | None:
    """Downloads audio from a URL using yt-dlp, extracts to WAV, and saves to output_dir."""
    print(f"Attempting to download audio from URL: {url}")
    try:
        info_ydl_opts = {'quiet': True, 'noplaylist': True, 'extract_flat': 'discard_in_extractor'}
        with yt_dlp.YoutubeDL(info_ydl_opts) as ydl_info:
            info = ydl_info.extract_info(url, download=False)
            video_title = info.get('title', f"untitled_video_{uuid.uuid4().hex[:8]}")
            sanitized_title_stem = sanitize_filename(video_title)
    except Exception as e:
        print(f"Could not extract video info for naming: {e}. Using a generic name.")
        sanitized_title_stem = f"downloaded_audio_{uuid.uuid4().hex[:8]}"

    output_filename_stem = output_dir / sanitized_title_stem
    expected_wav_path = output_filename_stem.with_suffix(".wav")

    ydl_opts_download = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_filename_stem),
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'noplaylist': True, 'quiet': False, 'progress': True, 'keepvideo': False, 'ignoreerrors': False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
            print(f"yt-dlp: Downloading and extracting audio. Expected output: {expected_wav_path}")
            error_code = ydl.download([url])
            if error_code != 0:
                print(f"Error: yt-dlp download process returned error code {error_code}.")
                return None
        if expected_wav_path.exists() and expected_wav_path.is_file():
            print(f"Audio successfully downloaded and extracted to: {expected_wav_path}")
            return expected_wav_path
        else:
            print(f"Warning: Expected file {expected_wav_path} not found. Searching in '{output_dir}' for stem '{sanitized_title_stem}.wav'...")
            for item in output_dir.iterdir():
                if item.stem == sanitized_title_stem and item.suffix.lower() == '.wav' and item.is_file():
                    print(f"Found matching WAV file: {item}")
                    return item
            print(f"Error: Could not reliably locate the downloaded WAV file in '{output_dir}'.")
            return None
    except yt_dlp.utils.DownloadError as de:
        print(f"yt-dlp DownloadError: {de}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_audio_from_local_video(video_path: Path, output_dir: Path) -> Path | None:
    """Extracts audio from a local video file."""
    if AudioFileClip is None:
        print("Error: 'moviepy' library is required to process local video files.")
        print("Please install it by running: pip install moviepy")
        return None
    print(f"Extracting audio from local video: {video_path.name}")
    try:
        # Ensure the output directory for extracted audio exists
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = output_dir / f"{video_path.stem}_extracted.wav"
        video_clip = AudioFileClip(str(video_path))
        video_clip.write_audiofile(str(output_filename))
        video_clip.close()
        print(f"Audio extracted successfully to: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error extracting audio from video '{video_path.name}': {e}")
        return None