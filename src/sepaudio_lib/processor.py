import shutil
import uuid
from pathlib import Path
import torch
import logging # For log_level constants

try:
    from audio_separator.separator import Separator
except ImportError:
    Separator = None 

try:
    # Updated import for SpeechBrain >= 1.0
    from speechbrain.inference.enhancement import SepformerEnhancement
    import torchaudio
    from torchaudio.transforms import Resample
except ImportError:
    SepformerEnhancement = None
    torchaudio = None
    Resample = None

def run_stage1_separation(input_audio_path: Path, output_dir: Path, model_name: str) -> Path | None:
    """Runs audio separation (Stage 1) and returns the path to the vocal stem."""
    if Separator is None:
        print("Error: 'audio-separator' library is not installed. Cannot perform separation.")
        print("Please run: pip install audio-separator[gpu] (or audio-separator for CPU)")
        return None

    print(f"\n--- Stage 1: Separating Dialogue using model '{model_name}' ---")
    print(f"Processing: {input_audio_path.name}")

    # Use a temporary subdirectory within the main output_dir for audio-separator's direct outputs
    temp_separation_subdir_name = f"audio_separator_temp_{input_audio_path.stem}_{uuid.uuid4().hex[:8]}"
    temp_separation_subdir = output_dir / temp_separation_subdir_name
    temp_separation_subdir.mkdir(parents=True, exist_ok=True)

    try:
        separator = Separator(
             output_dir=str(temp_separation_subdir), # audio-separator saves all its stems here
             log_level=logging.INFO, # Use the integer constant
             # model_file_dir=None, # Explicitly None to use default cache
             # denoise_output=True, # Example option, check library for more
        )
        print(f"Loading separation model '{model_name}'...")
        separator.load_model(model_name)
        
        print(f"Starting separation process for {input_audio_path}...")
        # The separate method might return a list of paths to all separated stems
        output_files_from_separator = separator.separate(str(input_audio_path))
        print(f"Separation process completed by audio-separator. Raw output files: {output_files_from_separator}")

        if not output_files_from_separator:
            print("Error: Separation produced no output files from audio-separator.")
            if temp_separation_subdir.exists(): shutil.rmtree(temp_separation_subdir)
            return None

        # Identify the vocal stem from the output files
        vocal_stem_path_in_temp_dir = None
        # Prioritize files explicitly named 'vocals' or matching primary stem output
        for f_path_str in output_files_from_separator:
            f_path = Path(f_path_str)
            # Common naming conventions: original_filename_vocals.wav or model-specific stem names
            if "vocals" in f_path.name.lower() or "voice" in f_path.name.lower():
                vocal_stem_path_in_temp_dir = f_path
                print(f"Identified vocal stem: {f_path.name}")
                break
        
        if not vocal_stem_path_in_temp_dir and output_files_from_separator: # Fallback
            vocal_stem_path_in_temp_dir = Path(output_files_from_separator[0]) # Assume primary stem
            print(f"Warning: Could not definitively find 'vocals' stem by name. Using primary output: {vocal_stem_path_in_temp_dir.name}")

        if vocal_stem_path_in_temp_dir and vocal_stem_path_in_temp_dir.exists():
            # Copy the identified vocal stem to the main output_dir with a standardized name
            final_stage1_output_path = output_dir / f"{input_audio_path.stem.replace('_extracted', '').replace('_inputcopy', '')}_S1_vocals.wav"
            shutil.copy(vocal_stem_path_in_temp_dir, final_stage1_output_path)
            print(f"Vocal stem (Stage 1) finalized to: {final_stage1_output_path}")
            
            # Clean up the temporary subdirectory used by audio-separator
            try:
                shutil.rmtree(temp_separation_subdir)
                print(f"Cleaned up temporary separation directory: {temp_separation_subdir}")
            except OSError as e:
                print(f"Warning: Could not remove temp separation dir {temp_separation_subdir}: {e}")
            return final_stage1_output_path
        else:
            print("Error: Vocal stem not found in separation output or separation failed to produce files.")
            if temp_separation_subdir.exists(): shutil.rmtree(temp_separation_subdir)
            return None

    except TypeError as te:
        print(f"TypeError during Separator initialization or usage: {te}")
        print("This often means the way Separator() or its methods are called doesn't match the installed library version, or log_level was a string instead of an integer (e.g., logging.INFO).")
        import traceback
        traceback.print_exc()
        if temp_separation_subdir.exists(): shutil.rmtree(temp_separation_subdir)
        return None
    except Exception as e:
        print(f"An error occurred during Stage 1 separation: {e}")
        import traceback
        traceback.print_exc()
        if temp_separation_subdir.exists(): shutil.rmtree(temp_separation_subdir)
        return None

def run_stage2_enhancement(vocals_audio_path: Path, output_dir: Path, model_name: str) -> Path | None:
    """Runs speech enhancement (Stage 2) on the separated vocal stem."""
    if SepformerEnhancement is None or torchaudio is None or Resample is None:
        print("Error: 'speechbrain' or 'torchaudio' not installed. Cannot perform enhancement.")
        return None

    print(f"\n--- Stage 2: Enhancing Dialogue using model '{model_name}' ---")
    print(f"Processing: {vocals_audio_path.name}")
    try:
        # Define a cache directory for SpeechBrain models within the output directory
        cache_dir_enhancer = output_dir / "pretrained_models_cache" / "speechbrain" / model_name.replace("/", "_")
        cache_dir_enhancer.mkdir(parents=True, exist_ok=True)

        enhancer = SepformerEnhancement.from_hparams(
            source=model_name,
            savedir=str(cache_dir_enhancer) # SpeechBrain will download/cache models here
        )
        
        target_sr = 16000 # Default for many SpeechBrain enhancers
        if "sepformer-dns4-16k" in model_name.lower(): target_sr = 16000
        elif "sepformer-dns4-8k" in model_name.lower(): target_sr = 8000
        # Add more specific model checks or make target_sr an argument if needed

        noisy_speech, original_sr = torchaudio.load(vocals_audio_path)

        if original_sr != target_sr:
            print(f"Resampling from {original_sr}Hz to {target_sr}Hz for enhancement model.")
            resampler = Resample(orig_freq=original_sr, new_freq=target_sr)
            noisy_speech = resampler(noisy_speech)
        
        # Ensure mono and correct shape (Batch, Time)
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

def finalize_audio_output(input_path: Path, output_path: Path, target_sample_rate: int | None, convert_to_mono: bool) -> Path | None:
    """Processes the final audio output: copies, optionally resamples, and converts to mono."""
    if torchaudio is None or Resample is None:
        print("Error: 'torchaudio' is required for resampling or mono conversion.")
        if input_path != output_path:
            try:
                shutil.copy(input_path, output_path)
                print(f"Copied as-is to: {output_path}")
                return output_path
            except Exception as copy_e:
                print(f"Error copying file during fallback: {copy_e}")
                return input_path # Return original if copy fails
        return input_path # Return original if no processing and paths are same

    print(f"\n--- Finalizing Output Audio: {input_path.name} ---")
    try:
        if not input_path.exists():
            print(f"Error: Input file for finalization not found: {input_path}")
            return None

        # If no processing is needed and paths are different, just copy
        if input_path != output_path and not target_sample_rate and not convert_to_mono:
            shutil.copy(input_path, output_path)
            print(f"Final output (copied as-is): {output_path}")
            return output_path
        # If paths are same and no processing, do nothing
        elif input_path == output_path and not target_sample_rate and not convert_to_mono:
            print(f"Final output (no changes needed): {output_path}")
            return output_path

        waveform, sr = torchaudio.load(input_path)

        did_process = False
        if convert_to_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print("Converted to mono.")
            did_process = True

        if target_sample_rate and sr != target_sample_rate:
            print(f"Resampling from {sr}Hz to {target_sample_rate}Hz...")
            resampler = Resample(orig_freq=sr, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sr = target_sample_rate # Update sample rate to the new one
            print(f"Resampled to {sr}Hz.")
            did_process = True

        if did_process or input_path != output_path : # Save if processed or if paths differ
            torchaudio.save(output_path, waveform, sr, format="wav")
            print(f"Final processed audio saved to: {output_path}")
        else: # Paths same, no processing
            print(f"Final audio (no changes made): {output_path}")

        return output_path

    except Exception as e:
        print(f"Error during audio finalization: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: copy original if processing fails and output_path is different
        if input_path != output_path:
            try:
                shutil.copy(input_path, output_path)
                print(f"Warning: Finalization failed. Copied original file to: {output_path}")
                return output_path
            except Exception as copy_e:
                print(f"Error copying original file during fallback: {copy_e}")
        return input_path # Return original if in-place or copy failed