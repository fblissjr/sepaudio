#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
from pathlib import Path
import re # For sanitizing filenames
import uuid # For unique filenames in download
from tqdm import tqdm # For progress bar

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
    # This is optional, checked later if local video processing is needed
    pass

try:
    # Updated SpeechBrain import for SpeechBrain 1.0+
    from speechbrain.inference.enhancement import SepformerEnhancement
    import torchaudio
    from torchaudio.transforms import Resample
except ImportError:
    # Optional, checked later if enhancement is enabled
    pass

# --- Helper Functions ---

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename component."""
    name = str(name) # Ensure it's a string
    name = re.sub(r'[^\w\s-]', '', name) # Remove invalid chars
    name = re.sub(r'[-\s]+', '-', name).strip('-') # Replace spaces/hyphens with single hyphen
    return name if name else "untitled"

def is_url(string: str) -> bool:
    """Basic check if a string is a URL."""
    s = str(string) # Ensure it's a string
    return s.startswith('http://') or s.startswith('https://') # or s.startswith('https:/') for a temporary hack

def download_audio_from_url(url: str, output_dir: Path) -> Path | None:
    """Downloads audio from a URL using yt-dlp, extracts to WAV, and saves to output_dir."""
    print(f"Attempting to download audio from URL: {url}")

    # First, extract info to get a title for our filename, and sanitize it
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
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'noplaylist': True,
        'quiet': False,
        'progress': True,
        'keepvideo': False,
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
            print(f"Warning: Expected file {expected_wav_path} not found. Searching in output directory...")
            for item in output_dir.iterdir():
                # Check if the stem matches (ignoring case for robustness if needed, but exact for now)
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
        print(f"Audio extracted successfully to: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error extracting audio from video '{video_path.name}': {e}")
        return None

def run_stage1_separation(input_audio_path: Path, output_dir: Path, model_name: str) -> Path | None:
    """Runs audio separation (Stage 1) and returns the path to the vocal stem."""
    print(f"\n--- Stage 1: Separating Dialogue using model '{model_name}' ---")
    print(f"Processing: {input_audio_path.name}")

    stage1_output_subdir_name = f"stage1_temp_{input_audio_path.stem}"
    stage1_output_subdir = output_dir / stage1_output_subdir_name
    stage1_output_subdir.mkdir(parents=True, exist_ok=True)

    try:
        separator = Separator(
            model_name=model_name,
            output_dir=str(stage1_output_subdir), # audio-separator saves its outputs here
            log_level='INFO',
        )

        output_files = separator.separate(str(input_audio_path))

        if not output_files:
            print("Error: Separation produced no output files.")
            return None

        vocal_stem_path_temp = None
        for f_path_str in output_files:
            f_path = Path(f_path_str)
            if "vocals" in f_path.name.lower() or input_audio_path.stem + "_" + model_name in f_path.name.lower(): # Common naming
                vocal_stem_path_temp = f_path
                break
        
        if not vocal_stem_path_temp and output_files: # Fallback
            vocal_stem_path_temp = Path(output_files[0]) # Assume primary is vocals
            print(f"Warning: Could not definitively find 'vocals' stem by name. Assuming first output is vocals: {vocal_stem_path_temp.name}")

        if vocal_stem_path_temp and vocal_stem_path_temp.exists():
            # Copy to a predictable name in the main output_dir
            final_stage1_output_path = output_dir / f"{input_audio_path.stem}_S1_vocals.wav"
            shutil.copy(vocal_stem_path_temp, final_stage1_output_path)
            print(f"Vocal stem (Stage 1) saved to: {final_stage1_output_path}")
            
            try: # Clean up temp subdir from audio-separator
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
        from speechbrain.inference.enhancement import SepformerEnhancement # Ensure up-to-date import
        import torchaudio
        from torchaudio.transforms import Resample
    except ImportError:
        print("Error: 'speechbrain' or 'torchaudio' not installed/found. Cannot perform enhancement.")
        print("Please run: pip install speechbrain torchaudio")
        return None

    print(f"\n--- Stage 2: Enhancing Dialogue using model '{model_name}' ---")
    print(f"Processing: {vocals_audio_path.name}")

    try:
        cache_dir_enhancer = output_dir / "pretrained_models_enhancer" / model_name.replace("/", "_")
        cache_dir_enhancer.mkdir(parents=True, exist_ok=True)

        enhancer = SepformerEnhancement.from_hparams(
            source=model_name,
            savedir=str(cache_dir_enhancer)
        )
        
        target_sr = 16000 # Common for many speechbrain models, esp. Sepformer-DNS
        if "sepformer-dns4-16k" in model_name: target_sr = 16000
        elif "sepformer-dns4-8k" in model_name: target_sr = 8000
        # Add more heuristics or make target_sr an argument if needed

        noisy_speech, original_sr = torchaudio.load(vocals_audio_path)

        if original_sr != target_sr:
            print(f"Resampling from {original_sr}Hz to {target_sr}Hz for enhancement model.")
            resampler = Resample(orig_freq=original_sr, new_freq=target_sr)
            noisy_speech = resampler(noisy_speech)
        
        if noisy_speech.ndim == 1: noisy_speech = noisy_speech.unsqueeze(0)
        if noisy_speech.shape[0] > 1: noisy_speech = torch.mean(noisy_speech, dim=0, keepdim=True)

        enhanced_speech = enhancer.enhance_batch(noisy_speech, lengths=torch.tensor([noisy_speech.shape[1]]))
        
        output_filename = output_dir / f"{vocals_audio_path.stem.replace('_S1_vocals', '')}_S2_enhanced.wav"
        torchaudio.save(output_filename, enhanced_speech.cpu().squeeze(0), target_sr)
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
    """Processes the final audio output: copies, optionally resamples, and converts to mono."""
    print(f"\n--- Finalizing Output Audio: {input_path.name} ---")
    try:
        if not input_path.exists():
            print(f"Error: Input file for finalization not found: {input_path}")
            return None

        if input_path == output_path and not target_sample_rate and not convert_to_mono:
            print(f"Final output (no changes needed): {output_path}")
            return output_path
        
        if target_sample_rate or convert_to_mono:
            try:
                import torchaudio
                from torchaudio.transforms import Resample
            except ImportError:
                print("Error: 'torchaudio' is required for resampling or mono conversion.")
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

        torchaudio.save(output_path, waveform, sr, format="wav") # Explicitly save as wav
        print(f"Final processed audio saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error during audio finalization: {e}")
        if input_path != output_path:
            try:
                shutil.copy(input_path, output_path)
                print(f"Warning: Finalization failed. Copied original file to: {output_path}")
                return output_path
            except Exception as copy_e:
                print(f"Error copying original file during fallback: {copy_e}")
        return input_path

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
        default=None,
        help="Resample the final dialogue output to this sample rate (e.g., 16000). Requires torchaudio."
    )
    parser.add_argument(
        "--output_mono",
        action="store_true",
        help="Convert the final dialogue output to mono. Requires torchaudio."
    )

    args = parser.parse_args()

    # --- DEBUG PRINT ---
    print(f"DEBUG: args.input_source value: '{args.input_source}'")
    print(f"DEBUG: type of args.input_source: {type(args.input_source)}")
    print(f"DEBUG: is_url(args.input_source) returns: {is_url(args.input_source)}")
    # --- END DEBUG PRINT ---

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_audio_file: Path | None = None
    temp_files_to_cleanup = []

    num_main_stages = 1 # Download/Input
    if args.separator_model_name: num_main_stages += 1 # Separation
    if args.enable_enhancement: num_main_stages += 1 # Enhancement
    num_main_stages +=1 # Finalization/Copy

    with tqdm(total=num_main_stages, desc="Overall Progress", unit="stage", dynamic_ncols=True) as pbar_main:
        pbar_main.set_description("Stage: Input/Download")
        if is_url(args.input_source):
            download_output_dir = output_dir / "downloaded_audio_temp"
            download_output_dir.mkdir(parents=True, exist_ok=True)
            current_audio_file = download_audio_from_url(args.input_source, download_output_dir)
            if current_audio_file and not args.keep_intermediate_files:
                temp_files_to_cleanup.append(current_audio_file)
        else:
            local_file_path = Path(args.input_source)
            if not local_file_path.exists():
                print(f"Error: Local file not found: {local_file_path}")
                sys.exit(1)
            
            video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']
            is_video_input = args.video or local_file_path.suffix.lower() in video_extensions

            if is_video_input:
                extract_output_dir = output_dir / "extracted_audio_temp"
                extract_output_dir.mkdir(parents=True, exist_ok=True)
                current_audio_file = extract_audio_from_local_video(local_file_path, extract_output_dir)
                if current_audio_file and not args.keep_intermediate_files:
                    temp_files_to_cleanup.append(current_audio_file)
            else:
                current_audio_file_name = f"{local_file_path.stem}_inputcopy{local_file_path.suffix}"
                current_audio_file = output_dir / current_audio_file_name
                shutil.copy(local_file_path, current_audio_file)
                print(f"Copied local audio to: {current_audio_file}")
                if not args.keep_intermediate_files:
                     temp_files_to_cleanup.append(current_audio_file)
        pbar_main.update(1)

        if not current_audio_file or not current_audio_file.exists():
            print("Error: Could not obtain a valid audio file to process.")
            sys.exit(1)

        pbar_main.set_description("Stage: Separation")
        stage1_vocals_path = run_stage1_separation(current_audio_file, output_dir, args.separator_model_name)
        pbar_main.update(1)

        if not stage1_vocals_path:
            print("Stage 1 separation failed. Exiting.")
            sys.exit(1)
        
        processed_audio_path_before_finalize = stage1_vocals_path
        if (args.enable_enhancement or args.output_sample_rate or args.output_mono) and not args.keep_intermediate_files:
            if stage1_vocals_path not in temp_files_to_cleanup:
                temp_files_to_cleanup.append(stage1_vocals_path)

        if args.enable_enhancement:
            pbar_main.set_description("Stage: Enhancement")
            stage2_enhanced_path = run_stage2_enhancement(stage1_vocals_path, output_dir, args.enhancer_model_name)
            pbar_main.update(1)
            if stage2_enhanced_path:
                processed_audio_path_before_finalize = stage2_enhanced_path
                if not args.keep_intermediate_files and stage1_vocals_path not in temp_files_to_cleanup:
                     temp_files_to_cleanup.append(stage1_vocals_path) # S1 is now intermediate
            else:
                print("Stage 2 enhancement failed. Using Stage 1 output for finalization.")
                if stage1_vocals_path in temp_files_to_cleanup and not (args.output_sample_rate or args.output_mono):
                    temp_files_to_cleanup.remove(stage1_vocals_path) # S1 is now final (before formatting)

        pbar_main.set_description("Stage: Finalization")
        base_name_for_final = Path(processed_audio_path_before_finalize.stem.replace("_S1_vocals", "").replace("_S2_enhanced", "").replace("_inputcopy",""))
        final_output_filename = output_dir / f"{base_name_for_final}_dialogue.wav"

        final_processed_path = finalize_audio_output(
            processed_audio_path_before_finalize,
            final_output_filename,
            args.output_sample_rate,
            args.output_mono
        )
        pbar_main.update(1)

        if final_processed_path != processed_audio_path_before_finalize and \
           processed_audio_path_before_finalize not in temp_files_to_cleanup and \
           not args.keep_intermediate_files:
            temp_files_to_cleanup.append(processed_audio_path_before_finalize)

    print("\n--- Processing Complete ---")
    if final_processed_path and final_processed_path.exists():
        print(f"Final processed dialogue track: {final_processed_path.resolve()}")
    else:
        print("No final output was successfully generated.")

    if not args.keep_intermediate_files:
        print("\nCleaning up intermediate files...")
        for f_path in set(temp_files_to_cleanup):
            if f_path and f_path.exists() and (not final_processed_path or f_path.resolve() != final_processed_path.resolve()):
                try:
                    f_path.unlink()
                    print(f"  - Removed: {f_path.name}")
                except OSError as e:
                    print(f"Warning: Could not remove {f_path.name}: {e}")
        
        for subdirname in ["downloaded_audio_temp", "extracted_audio_temp", f"stage1_temp_{current_audio_file.stem if current_audio_file else 'unknown'}"]:
            subdir = output_dir / subdirname
            try:
                if subdir.exists() and not any(subdir.iterdir()):
                    shutil.rmtree(subdir)
                    print(f"  - Removed empty directory: {subdir.name}")
            except OSError as e:
                 print(f"Warning: Could not remove empty directory {subdir.name}: {e}") # Can happen if it was already removed (e.g. stage1_temp)
    else:
        print(f"\nIntermediate files kept in subdirectories of: {output_dir.resolve()}")

if __name__ == "__main__":
    main()