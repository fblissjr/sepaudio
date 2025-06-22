#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
from pathlib import Path
import uuid
import re
from tqdm import tqdm

# --- Dependency Import and Checks ---
try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp is not installed. Please run: pip install yt-dlp")
    sys.exit(1)

try:
    from audio_separator.separator import Separator
    import torch # audio-separator depends on PyTorch
except ImportError:
    print("Error: 'audio-separator' or 'torch' not installed. Please run: pip install audio-separator[gpu] (recommended) or pip install audio-separator")
    print("Ensure PyTorch is correctly installed, especially with CUDA support for GPU.")
    sys.exit(1)

try:
    from moviepy.editor import AudioFileClip
except ImportError:
    pass # Optional, checked later

try:
    from speechbrain.pretrained import SepformerEnhancement # Example
    import torchaudio
    from torchaudio.transforms import Resample
except ImportError:
    pass # Optional, checked later

# --- Helper Functions ---
def download_audio_from_url(url: str, output_dir: Path) -> Path | None:
    """Downloads audio from a URL using yt-dlp, extracts to WAV, and saves to output_dir."""
    print(f"Attempting to download audio from URL: {url}")

    # First, extract info to get a title for our filename, and sanitize it
    try:
        # Use a temporary ydl instance just to get info without full logging if not needed
        info_ydl_opts = {'quiet': True, 'noplaylist': True, 'extract_flat': 'discard_in_extractor'}
        with yt_dlp.YoutubeDL(info_ydl_opts) as ydl_info:
            info = ydl_info.extract_info(url, download=False)
            # Use a UUID if title is missing or too generic to avoid collisions
            video_title = info.get('title', f"untitled_video_{uuid.uuid4().hex[:8]}")
            sanitized_title_stem = sanitize_filename(video_title)
    except Exception as e:
        print(f"Could not extract video info for naming: {e}. Using a generic name.")
        sanitized_title_stem = f"downloaded_audio_{uuid.uuid4().hex[:8]}"

    # Define the exact output path stem (yt-dlp's audio extractor will add .wav)
    # This is where the final .wav file should appear.
    output_filename_stem = output_dir / sanitized_title_stem
    expected_wav_path = output_filename_stem.with_suffix(".wav")

    ydl_opts_download = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_filename_stem),  # Output stem, postprocessor adds extension
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192', # Not directly for WAV, but good practice if intermediate is used
        }],
        'noplaylist': True,
        'quiet': False, # Show yt-dlp's own progress
        'progress': True,
        'keepvideo': False, # Delete intermediate downloaded file (e.g. webm, mp4)
        'ignoreerrors': False,
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
            # Fallback if the exact filename isn't found (e.g. due to yt-dlp subtle sanitization differences)
            print(f"Warning: Expected file {expected_wav_path} not found. Searching in output directory...")
            for item in output_dir.iterdir():
                if item.stem == sanitized_title_stem and item.suffix.lower() == '.wav' and item.is_file():
                    print(f"Found matching WAV file: {item}")
                    return item
            print(f"Error: Could not reliably locate the downloaded WAV file in '{output_dir}'. Please check yt-dlp's output above.")
            return None

    except yt_dlp.utils.DownloadError as de:
        print(f"yt-dlp DownloadError: {de}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        import traceback
        traceback.print_exc()
        return None

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename component."""
    name = re.sub(r'[^\w\s-]', '', name) # Remove invalid chars
    name = re.sub(r'[-\s]+', '-', name).strip('-') # Replace spaces/hyphens with single hyphen
    return name if name else "untitled"

def is_url(string: str) -> bool:
    """Basic check if a string is a URL."""
    return string.startswith('http://') or string.startswith('https://')

def download_audio_from_url(url: str, output_dir: Path) -> Path | None:
    """Downloads audio from a URL using yt-dlp, saves to output_dir."""
    print(f"Attempting to download audio from URL: {url}")
    download_path_template = output_dir / "downloaded_audio_%(title)s.%(ext)s"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(download_path_template),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav', # Output as WAV for consistency
            'preferredquality': '192',
        }],
        'noplaylist': True,
        'quiet': False, # Set to True for less output
        'progress': True,
    }

    downloaded_file_path = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False) # Get info first
            title = sanitize_filename(info_dict.get('title', 'untitled'))
            
            # Construct a more predictable output filename
            # yt-dlp's templating in outtmpl for preferredcodec doesn't always yield expected ext
            # So, we'll download, then rename if necessary, or just rely on a known pattern.
            # For simplicity, we'll let yt-dlp name it and then find it.
            # A more robust way would be to hook into yt-dlp's progress hooks to get the exact filename.

            # Simplified: Use a fixed stem for the downloaded file before conversion
            temp_download_name = f"yt_dlp_temp_download_{title}"
            specific_outtmpl = output_dir / f"{temp_download_name}.%(ext)s"
            ydl_opts_download = ydl_opts.copy()
            ydl_opts_download['outtmpl'] = str(specific_outtmpl)

            print(f"Downloading to template: {specific_outtmpl}")
            ydl.download([url]) # Actual download with modified opts

            # Find the downloaded .wav file (yt-dlp should convert it)
            # This part can be tricky if yt-dlp's output naming is complex
            # We assume it creates ONE .wav file based on our temp_download_name
            for item in output_dir.iterdir():
                if item.is_file() and item.name.startswith(temp_download_name) and item.suffix.lower() == '.wav':
                    downloaded_file_path = item
                    break
            
            if not downloaded_file_path: # Fallback if specific naming failed
                 for item in output_dir.iterdir(): # Broad search for a .wav file with title
                    if title in item.name and item.suffix.lower() == '.wav':
                        downloaded_file_path = item
                        break

            if downloaded_file_path:
                print(f"Audio downloaded and converted to WAV: {downloaded_file_path}")
                return downloaded_file_path
            else:
                print(f"Error: Could not locate the downloaded WAV file in {output_dir}. Check yt-dlp output.")
                return None

    except Exception as e:
        print(f"Error downloading audio from {url}: {e}")
        return None

def extract_audio_from_local_video(video_path: Path, output_dir: Path) -> Path | None:
    """Extracts audio from a local video file."""
    try:
        from moviepy.editor import AudioFileClip # Ensure available
    except ImportError:
        print("Error: 'moviepy' library is required to process local video files.")
        print("Please install it by running: pip install moviepy")
        return None

    print(f"Extracting audio from local video: {video_path.name}")
    try:
        output_filename = output_dir / f"{video_path.stem}_extracted.wav"
        video_clip = AudioFileClip(str(video_path))
        video_clip.write_audiofile(str(output_filename))
        video_clip.close()
        print(f"Audio extracted to: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error extracting audio from video '{video_path.name}': {e}")
        return None

def run_stage1_separation(input_audio_path: Path, output_dir: Path, model_name: str) -> Path | None:
    """Runs audio separation (Stage 1) and returns the path to the vocal stem."""
    print(f"\n--- Stage 1: Separating Dialogue using model '{model_name}' ---")
    print(f"Processing: {input_audio_path.name}")
    
    # Configure separator
    # output_dir for Separator is where it puts its *own* output files.
    # The library will create filenames based on input name and model.
    stage1_output_subdir = output_dir / "stage1_separated_temp"
    stage1_output_subdir.mkdir(parents=True, exist_ok=True)

    try:
        separator = Separator(
            model_name=model_name,
            output_dir=str(stage1_output_subdir),
            log_level='INFO',
            # For GPU: audio-separator usually auto-detects if PyTorch is set up for CUDA
            # use_cuda=torch.cuda.is_available(), # Can be explicit
        )
        
        output_files = separator.separate(str(input_audio_path))

        if not output_files:
            print("Error: Separation produced no output files.")
            return None

        # Find the vocal stem. This is model-dependent.
        # For many vocal models, it's the primary stem. For Demucs, it's named 'vocals'.
        vocal_stem_path = None
        for f_path_str in output_files:
            f_path = Path(f_path_str)
            # Common naming conventions: often includes "vocals", "Vocals", or is the primary stem.
            if "vocals" in f_path.name.lower() or "voice" in f_path.name.lower():
                vocal_stem_path = f_path
                break
        
        if not vocal_stem_path and output_files: # Fallback: assume primary stem is vocals if not found by name
            vocal_stem_path = Path(output_files[0])
            print(f"Warning: Could not definitively find 'vocals' stem by name. Assuming first output is vocals: {vocal_stem_path.name}")


        if vocal_stem_path and vocal_stem_path.exists():
            # Copy to a predictable name in the main output_dir
            final_stage1_output_path = output_dir / f"{input_audio_path.stem}_S1_vocals.wav"
            shutil.copy(vocal_stem_path, final_stage1_output_path)
            print(f"Vocal stem (Stage 1) saved to: {final_stage1_output_path}")
            
            # Clean up temp subdir from audio-separator
            try:
                shutil.rmtree(stage1_output_subdir)
            except OSError as e:
                print(f"Warning: Could not remove temp separation dir {stage1_output_subdir}: {e}")
            return final_stage1_output_path
        else:
            print("Error: Vocal stem not found or separation failed.")
            return None

    except Exception as e:
        print(f"Error during Stage 1 separation: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_stage2_enhancement(vocals_audio_path: Path, output_dir: Path, model_name: str) -> Path | None:
    """Runs speech enhancement (Stage 2) on the separated vocal stem."""
    try:
        from speechbrain.pretrained import SepformerEnhancement # Ensure available
        import torchaudio
        from torchaudio.transforms import Resample
    except ImportError:
        print("Error: 'speechbrain' or 'torchaudio' not installed/found. Cannot perform enhancement.")
        print("Please run: pip install speechbrain torchaudio")
        return None

    print(f"\n--- Stage 2: Enhancing Dialogue using model '{model_name}' ---")
    print(f"Processing: {vocals_audio_path.name}")

    try:
        # Load pre-trained enhancement model
        # Note: Model source might vary. This is an example.
        # Some models require specific sample rates (e.g., 16kHz for sepformer-dns4)
        enhancer = SepformerEnhancement.from_hparams(
            source=model_name, # e.g., "speechbrain/sepformer-dns4-16k-enhancement"
            savedir=output_dir / "pretrained_models" / model_name.replace("/", "_") # Cache dir for models
        )
        
        # Determine model's expected sample rate (often in its hparams or docs)
        # For sepformer-dns4-16k-enhancement, it's 16000 Hz.
        # This part might need to be more robust if using various enhancer models.
        # For this example, assuming 16kHz for the default model.
        target_sr = 16000
        if "16k" not in model_name and "8k" in model_name: # Basic heuristic
            target_sr = 8000
        
        # Load the separated vocal track
        noisy_speech, original_sr = torchaudio.load(vocals_audio_path)

        # Resample if necessary
        if original_sr != target_sr:
            print(f"Resampling from {original_sr}Hz to {target_sr}Hz for enhancement model.")
            resampler = Resample(orig_freq=original_sr, new_freq=target_sr)
            noisy_speech = resampler(noisy_speech)
        
        # Ensure mono and correct shape for SpeechBrain (Batch, Time, Channels - or Batch, Time)
        if noisy_speech.ndim == 1:
            noisy_speech = noisy_speech.unsqueeze(0) # Add batch dimension
        if noisy_speech.shape[0] > 1 and noisy_speech.ndim == 2 : # if stereo (Batch, Channels, Time) -> (Batch, Time)
             noisy_speech = noisy_speech.mean(dim=1) # Average channels for mono
             if noisy_speech.ndim == 1: noisy_speech = noisy_speech.unsqueeze(0)


        # Enhance (SpeechBrain expects Batch, Time for enhance_batch)
        enhanced_speech = enhancer.enhance_batch(noisy_speech, lengths=torch.tensor([noisy_speech.shape[1]]))
        
        # Save the enhanced speech
        output_filename = output_dir / f"{vocals_audio_path.stem.replace('_S1_vocals', '')}_S2_enhanced.wav"
        torchaudio.save(output_filename, enhanced_speech.cpu().squeeze(0), target_sr) # Save at target_sr
        print(f"Enhanced vocals (Stage 2) saved to: {output_filename}")
        return output_filename

    except Exception as e:
        print(f"Error during Stage 2 enhancement: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def finalize_audio_output(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int | None,
    convert_to_mono: bool,
) -> Path | None:
    """
    Processes the final audio output: copies, optionally resamples, and converts to mono.
    Returns the path to the processed file.
    """
    print(f"\n--- Finalizing Output Audio: {input_path.name} ---")
    try:
        if not input_path.exists():
            print(f"Error: Input file for finalization not found: {input_path}")
            return None

        # If no processing needed other than copying to a final name
        if input_path == output_path and not target_sample_rate and not convert_to_mono:
            print(f"Final output (no changes): {output_path}")
            return output_path
        
        # Ensure torchaudio is available if processing is needed
        if target_sample_rate or convert_to_mono:
            try:
                import torchaudio
                from torchaudio.transforms import Resample
            except ImportError:
                print("Error: 'torchaudio' is required for resampling or mono conversion.")
                print("If no resampling/mono conversion is needed, avoid --output_sample_rate and --output_mono.")
                print(f"Copying as-is to: {output_path}")
                if input_path != output_path: shutil.copy(input_path, output_path)
                return output_path

        waveform, sr = torchaudio.load(input_path)

        if convert_to_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print("Converted to mono.")

        if target_sample_rate and sr != target_sample_rate:
            print(f"Resampling from {sr}Hz to {target_sample_rate}Hz...")
            resampler = Resample(orig_freq=sr, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sr = target_sample_rate
            print(f"Resampled to {sr}Hz.")

        torchaudio.save(output_path, waveform, sr)
        print(f"Final processed audio saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error during audio finalization: {e}")
        # Fallback: copy original if processing fails but output_path is different
        if input_path != output_path:
            try:
                shutil.copy(input_path, output_path)
                print(f"Warning: Finalization failed. Copied original file to: {output_path}")
                return output_path
            except Exception as copy_e:
                print(f"Error copying original file during fallback: {copy_e}")
        return input_path # Return original if in-place or copy failed

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Advanced Dialogue Processor: Downloads, separates, and enhances dialogue from audio/video.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_source",
        help="Path to a local audio/video file OR a URL (e.g., YouTube video)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="processed_dialogue_output",
        help="Directory to save all output files (default: processed_dialogue_output)."
    )
    parser.add_argument(
        "--separator_model_name",
        default="UVR-MDX-NET-Voc_FT",
        help="Model name for Stage 1 audio separation (from 'audio-separator' library).\n"
             "Examples: 'UVR-MDX-NET-Voc_FT', 'htdemucs_ft', 'MDX_NET_Inst_HQ_3'.\n"
             "Refer to 'audio-separator --list_models' or library docs."
    )
    parser.add_argument(
        "--enable_enhancement",
        action="store_true",
        help="Enable Stage 2 speech enhancement on the separated vocals."
    )
    parser.add_argument(
        "--enhancer_model_name",
        default="speechbrain/sepformer-dns4-16k-enhancement",
        help="Model name for Stage 2 speech enhancement (from SpeechBrain/Hugging Face).\n"
             "Example: 'speechbrain/sepformer-dns4-16k-enhancement' (often expects 16kHz input)."
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Explicitly treat a local input file as video (if extension isn't standard video type)."
    )
    parser.add_argument(
        "--keep_intermediate_files",
        action="store_true",
        help="Keep all intermediate files (downloaded audio, extracted audio, Stage 1 vocals before enhancement)."
    )
    parser.add_argument(
        "--output_sample_rate",
        type=int,
        default=None, # No resampling by default
        help="Resample the final dialogue output to this sample rate (e.g., 16000). Requires torchaudio."
    )
    parser.add_argument(
        "--output_mono",
        action="store_true",
        help="Convert the final dialogue output to mono. Requires torchaudio."
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_audio_file: Path | None = None
    temp_files_to_cleanup = [] # Store paths of temporary files created by this script

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_audio_file: Path | None = None
    files_to_cleanup = []

    # Determine number of main stages for tqdm
    num_stages = 1 # Download/Input
    if args.separator_model_name: num_stages += 1 # Separation
    if args.enable_enhancement: num_stages += 1 # Enhancement
    if args.output_sample_rate or args.output_mono: num_stages +=1 # Finalization (if distinct from last step)
    else: num_stages +=1 # For final copy/naming if no other processing

    with tqdm(total=num_stages, desc="Overall Progress", unit="stage", dynamic_ncols=True) as pbar:
        # --- Step 0: Input Handling ---
        local_file_path = Path(args.input_source)
        if not local_file_path.exists():
            print(f"Error: Local file not found: {local_file_path}")
            sys.exit(1)
        
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']
        is_video_input = args.video or local_file_path.suffix.lower() in video_extensions

        if is_video_input:
            extract_output_dir = output_dir / "extracted_video_audio"
            extract_output_dir.mkdir(parents=True, exist_ok=True)
            current_audio_file = extract_audio_from_local_video(local_file_path, extract_output_dir)
            if current_audio_file and not args.keep_intermediate_files:
                temp_files_to_cleanup.append(current_audio_file)
        else:
            # For local audio, copy to output_dir for consistent processing path
            # The original local audio is not a "temp" file created by this script.
            current_audio_file_name = f"{local_file_path.stem}_input_copy{local_file_path.suffix}"
            current_audio_file = output_dir / current_audio_file_name
            shutil.copy(local_file_path, current_audio_file)
            print(f"Copied local audio to: {current_audio_file}")
            if not args.keep_intermediate_files: # This copy is intermediate
                 temp_files_to_cleanup.append(current_audio_file)


    if not current_audio_file or not current_audio_file.exists():
        print("Error: Could not obtain a valid audio file to process.")
        sys.exit(1)

    # --- Stage 1: Separation ---
    # Note: run_stage1_separation already saves its primary output (vocals) to output_dir
    # with a name like f"{input_audio_path.stem}_S1_vocals.wav"
    stage1_vocals_path = run_stage1_separation(current_audio_file, output_dir, args.separator_model_name)

    if not stage1_vocals_path:
        print("Stage 1 separation failed. Exiting.")
        sys.exit(1)
    
    # This S1 output is intermediate if enhancement is enabled OR if final processing changes it
    if (args.enable_enhancement or args.output_sample_rate or args.output_mono) and not args.keep_intermediate_files:
        temp_files_to_cleanup.append(stage1_vocals_path)

    processed_audio_path_before_finalize = stage1_vocals_path

    # --- Stage 2: Enhancement (Optional) ---
    if args.enable_enhancement:
        stage2_enhanced_path = run_stage2_enhancement(stage1_vocals_path, output_dir, args.enhancer_model_name)
        if stage2_enhanced_path:
            processed_audio_path_before_finalize = stage2_enhanced_path
            # If S2 succeeded and we don't keep intermediates, S1 output is definitely temp.
            # (It was already added to cleanup if S2 is enabled)
        else:
            print("Stage 2 enhancement failed. Using Stage 1 output for finalization.")
            # If S2 failed, S1 output is what we will finalize. It shouldn't be in temp_files_to_cleanup
            # if keep_intermediate_files is False UNLESS finalization changes it.
            if stage1_vocals_path in temp_files_to_cleanup and not (args.output_sample_rate or args.output_mono):
                temp_files_to_cleanup.remove(stage1_vocals_path)


    # --- Stage 3: Final Output Processing (Resample/Mono) ---
    # Define the final filename
    base_name_for_final = Path(processed_audio_path_before_finalize.stem.replace("_S1_vocals", "").replace("_S2_enhanced", ""))
    final_output_filename = output_dir / f"{base_name_for_final}_final_dialogue.wav"

    final_processed_path = finalize_audio_output(
        processed_audio_path_before_finalize,
        final_output_filename,
        args.output_sample_rate,
        args.output_mono
    )

    # If finalization created a new file and the pre-finalized one is different,
    # and we are not keeping intermediates, the pre-finalized one is temp.
    if final_processed_path != processed_audio_path_before_finalize and \
       processed_audio_path_before_finalize not in temp_files_to_cleanup and \
       not args.keep_intermediate_files:
        temp_files_to_cleanup.append(processed_audio_path_before_finalize)


    print("\n--- Processing Complete ---")
    if final_processed_path and final_processed_path.exists():
        print(f"Final processed dialogue track: {final_processed_path.resolve()}")
    else:
        print("No final output was successfully generated.")

    # --- Cleanup ---
    if not args.keep_intermediate_files:
        print("\nCleaning up intermediate files...")
        for f_path in set(temp_files_to_cleanup): # Use set to avoid duplicates
            if f_path and f_path.exists() and f_path != final_processed_path: # Don't delete the final output
                try:
                    f_path.unlink()
                    print(f"  - Removed: {f_path.name}")
                except OSError as e:
                    print(f"Warning: Could not remove {f_path.name}: {e}")
        
        # Clean up empty intermediate directories created by this script
        for subdirname in ["downloaded_audio_files", "extracted_video_audio"]:
            subdir = output_dir / subdirname
            try:
                if subdir.exists() and not any(subdir.iterdir()): # Check if empty
                    shutil.rmtree(subdir)
                    print(f"  - Removed empty directory: {subdir.name}")
            except OSError: pass # Ignore
    else:
        print(f"\nIntermediate files kept in subdirectories of: {output_dir.resolve()}")


if __name__ == "__main__":
    main()