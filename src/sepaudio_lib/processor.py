import shutil
import uuid
from pathlib import Path
import torch # Keep torch import for type hints if used, or for general utility

try:
    from audio_separator.separator import Separator
except ImportError:
    Separator = None # Handle missing optional dependency

try:
    from speechbrain.inference.enhancement import SepformerEnhancement
    import torchaudio
    from torchaudio.transforms import Resample
except ImportError:
    SepformerEnhancement = None # Handle missing optional dependency
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

    temp_separation_subdir_name = f"audio_separator_temp_{input_audio_path.stem}_{uuid.uuid4().hex[:8]}"
    temp_separation_subdir = output_dir / temp_separation_subdir_name
    temp_separation_subdir.mkdir(parents=True, exist_ok=True)

    try:
        # This is the part that was causing the error.
        # Let's try the MOST basic initialization and then configuration.
        # separator = Separator() # Try initializing with no arguments
        # separator.output_dir = str(temp_separation_subdir)
        # separator.log_level = 'INFO'
        # separator.load_model(model_name=model_name)

        # Alternative based on some library patterns if the above still fails:
        # Directly from the library's own CLI example structure if it's different
        # For now, assuming the common API pattern:
        separator = Separator(
             output_dir=str(temp_separation_subdir), # Usually okay
             log_level='INFO'
        )
        separator.load_model(model_name) # Pass model_name here


        print(f"Model '{model_name}' loaded for separation.")
        print(f"Starting separation process for {input_audio_path}...")
        output_files = separator.separate(str(input_audio_path))
        print(f"Separation process completed. Raw output files: {output_files}")

        if not output_files:
            print("Error: Separation produced no output files from audio-separator.")
            shutil.rmtree(temp_separation_subdir)
            return None

        vocal_stem_path_in_temp_dir = None
        for f_path_str in output_files:
            f_path = Path(f_path_str)
            if "vocals" in f_path.name.lower():
                vocal_stem_path_in_temp_dir = f_path
                print(f"Identified vocal stem: {f_path.name}")
                break
        if not vocal_stem_path_in_temp_dir and output_files:
            vocal_stem_path_in_temp_dir = Path(output_files[0])
            print(f"Warning: Using primary output as vocal stem: {vocal_stem_path_in_temp_dir.name}")

        if vocal_stem_path_in_temp_dir and vocal_stem_path_in_temp_dir.exists():
            final_stage1_output_path = output_dir / f"{input_audio_path.stem}_S1_vocals.wav"
            shutil.copy(vocal_stem_path_in_temp_dir, final_stage1_output_path)
            print(f"Vocal stem (Stage 1) finalized to: {final_stage1_output_path}")
            try:
                shutil.rmtree(temp_separation_subdir)
                print(f"Cleaned up temporary separation directory: {temp_separation_subdir}")
            except OSError as e:
                print(f"Warning: Could not remove temp separation dir {temp_separation_subdir}: {e}")
            return final_stage1_output_path
        else:
            print("Error: Vocal stem not found or separation failed.")
            if temp_separation_subdir.exists(): shutil.rmtree(temp_separation_subdir)
            return None
    except TypeError as te: # Catch the specific TypeError again
        print(f"TypeError during Separator initialization or usage: {te}")
        print("This often means the way Separator() or its methods are called doesn't match the installed library version.")
        print("Please check the audio-separator library's documentation for correct API usage.")
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
        cache_dir_enhancer = output_dir / "pretrained_models_enhancer" / model_name.replace("/", "_")
        cache_dir_enhancer.mkdir(parents=True, exist_ok=True)
        enhancer = SepformerEnhancement.from_hparams(source=model_name, savedir=str(cache_dir_enhancer))
        target_sr = 16000
        if "sepformer-dns4-16k" in model_name: target_sr = 16000
        elif "sepformer-dns4-8k" in model_name: target_sr = 8000
        noisy_speech, original_sr = torchaudio.load(vocals_audio_path)
        if original_sr != target_sr:
            print(f"Resampling from {original_sr}Hz to {target_sr}Hz.")
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

def finalize_audio_output(input_path: Path, output_path: Path, target_sample_rate: int | None, convert_to_mono: bool) -> Path | None:
    """Processes the final audio output: copies, optionally resamples, and converts to mono."""
    if torchaudio is None or Resample is None:
        print("Error: 'torchaudio' is required for resampling or mono conversion.")
        if input_path != output_path: shutil.copy(input_path, output_path)
        print(f"Copied as-is to: {output_path}")
        return output_path if input_path != output_path else input_path

    print(f"\n--- Finalizing Output Audio: {input_path.name} ---")
    try:
        if not input_path.exists():
            print(f"Error: Input file for finalization not found: {input_path}")
            return None
        if input_path == output_path and not target_sample_rate and not convert_to_mono:
            print(f"Final output (no changes needed): {output_path}")
            return output_path
        waveform, sr = torchaudio.load(input_path)
        if convert_to_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print("Converted to mono.")
        if target_sample_rate and sr != target_sample_rate:
            print(f"Resampling from {sr}Hz to {target_sample_rate}Hz.")
            resampler = Resample(orig_freq=sr, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sr = target_sample_rate
        torchaudio.save(output_path, waveform, sr, format="wav")
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