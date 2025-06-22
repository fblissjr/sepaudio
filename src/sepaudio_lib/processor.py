# src/sepaudio_lib/processor.py
import shutil
import uuid
from pathlib import Path
import torch
import logging  # For log_level constants

# Attempt to import necessary libraries, allowing for them to be optional
try:
    from audio_separator.separator import Separator
except ImportError:
    Separator = None

try:
    # Updated import for SpeechBrain >= 1.0 for speech enhancement tasks
    import torchaudio
    from speechbrain.inference.enhancement import SepformerEnhancement
    from torchaudio.transforms import Resample
except ImportError:
    SepformerEnhancement = None  # Mark as unavailable if import fails
    torchaudio = None
    Resample = None


def run_stage1_separation(input_audio_path: Path, output_dir: Path, model_name: str) -> Path | None:
    """
    Performs Stage 1 audio separation (e.g., vocals from music).
    It uses the 'audio-separator' library to process the input audio.
    Models are loaded by 'audio-separator', which should handle downloading if needed.
    The identified vocal stem is copied to the main output directory.

    Args:
        input_audio_path: Path to the input audio file.
        output_dir: The primary directory where final outputs should be stored.
                    A temporary subdirectory will be created here for raw separation outputs.
        model_name: The name of the separation model to use (e.g., 'UVR-MDX-NET-Voc_FT.onnx').

    Returns:
        Path to the separated vocal stem if successful, else None.
    """
    if Separator is None:
        print("Error: 'audio-separator' library is not installed. Cannot perform separation.")
        print("Please run: pip install audio-separator[gpu] (or audio-separator for CPU)")
        return None

    print(f"\n--- Stage 1: Separating Dialogue using model '{model_name}' ---")
    print(f"Processing: {input_audio_path.name}")

    # Create a unique temporary subdirectory within output_dir for this specific separation task's raw outputs.
    # This keeps the main output_dir clean and helps manage intermediate files from audio-separator.
    temp_separation_subdir_name = f"audio_separator_temp_{input_audio_path.stem}_{uuid.uuid4().hex[:8]}"
    temp_separation_subdir = output_dir / temp_separation_subdir_name
    temp_separation_subdir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize the Separator.
        # output_dir for Separator is where it will save all its stems (vocals, instrumental, etc.).
        # log_level is set to INFO using the logging module's constant.
        # model_file_dir is deliberately omitted to allow audio-separator to use its
        # default model caching/downloading mechanism (typically to a user cache like ~/.cache).
        print(
            f"DEBUG (processor.py): Initializing Separator(output_dir='{str(temp_separation_subdir)}', log_level=logging.INFO) - OMITTING model_file_dir"
        )
        separator = Separator(output_dir=str(temp_separation_subdir), log_level=logging.INFO)

        # Log the directory audio-separator is actually using for its models.
        # This helps debug if it's not using the expected default cache.
        actual_model_dir_used = getattr(
            separator, "model_file_dir", "Attribute 'model_file_dir' not found or None"
        )
        if (
            actual_model_dir_used == "Attribute 'model_file_dir' not found or None"
        ):  # Try other common attribute names
            actual_model_dir_used = getattr(separator, "cache_dir", "Attribute 'cache_dir' not found or None")
        print(
            f"DEBUG (processor.py): Separator instance will attempt to use model directory: {actual_model_dir_used}"
        )

        print(f"Loading separation model '{model_name}'...")
        separator.load_model(model_name)  # Load the specified model.

        print(f"Starting separation process for {input_audio_path}...")
        # The separate method returns a list of filenames relative to the Separator's output_dir.
        output_filenames_from_separator = separator.separate(str(input_audio_path))
        print(
            f"Separation process completed by audio-separator. Raw output file (names): {output_filenames_from_separator}"
        )

        if not output_filenames_from_separator:
            print("Error: Separation produced no output files from audio-separator.")
            if temp_separation_subdir.exists():
                shutil.rmtree(temp_separation_subdir)
            return None

        # Identify the vocal stem from the list of output filenames.
        identified_vocal_stem_full_path = None
        for relative_filename in output_filenames_from_separator:
            # Construct the full path to where audio-separator saved the file.
            full_path_to_stem = temp_separation_subdir / relative_filename
            if "vocals" in full_path_to_stem.name.lower() or "voice" in full_path_to_stem.name.lower():
                identified_vocal_stem_full_path = full_path_to_stem
                print(f"Identified vocal stem (full path): {identified_vocal_stem_full_path}")
                break

        # Fallback: if no file explicitly named "vocals" is found, assume the first output is the primary desired stem.
        if not identified_vocal_stem_full_path and output_filenames_from_separator:
            identified_vocal_stem_full_path = temp_separation_subdir / output_filenames_from_separator[0]
            print(
                f"Warning: Using primary output as vocal stem (full path): {identified_vocal_stem_full_path}"
            )

        # Check if the identified vocal stem exists and then copy it to the main output directory.
        if identified_vocal_stem_full_path and identified_vocal_stem_full_path.exists():
            # Create a clean, standardized name for the vocal stem in the main output directory.
            clean_input_stem = input_audio_path.stem.replace("_extracted", "").replace("_inputcopy", "")
            final_stage1_output_path = output_dir / f"{clean_input_stem}_S1_vocals.wav"
            shutil.copy(identified_vocal_stem_full_path, final_stage1_output_path)
            print(f"Vocal stem (Stage 1) finalized to: {final_stage1_output_path}")

            # Clean up the temporary subdirectory created by audio-separator for its raw outputs.
            # This is done regardless of keep_intermediate_files for this specific temp dir,
            # as the main desired output (vocals) has been copied out.
            try:
                shutil.rmtree(temp_separation_subdir)
                print(f"Cleaned up temporary separation work directory: {temp_separation_subdir}")
            except OSError as e:
                print(f"Warning: Could not remove temp separation work dir {temp_separation_subdir}: {e}")
            return final_stage1_output_path
        else:
            # Log error if the identified stem path doesn't exist or no stem was identified.
            if identified_vocal_stem_full_path:
                print(f"Error: Identified vocal stem '{identified_vocal_stem_full_path}' does not exist.")
            else:
                print("Error: No vocal stem could be identified from separation output list.")

            if temp_separation_subdir.exists():
                shutil.rmtree(temp_separation_subdir)  # Cleanup on failure
            return None

    except TypeError as te:
        print(f"TypeError during Separator initialization or usage in processor.py: {te}")
        print(
            "This typically indicates an issue with arguments to Separator() or its methods (e.g., log_level type)."
        )
        import traceback

        traceback.print_exc()
        if temp_separation_subdir.exists():
            shutil.rmtree(temp_separation_subdir)
        return None
    except ValueError as ve:  # Catch ValueError which audio-separator uses for "model not found"
        print(f"ValueError during Stage 1 separation in processor.py (model: {model_name}): {ve}")
        print("This often means the model name is incorrect or model files could not be downloaded/found.")
        import traceback

        traceback.print_exc()
        if temp_separation_subdir.exists():
            shutil.rmtree(temp_separation_subdir)
        return None
    except Exception as e:  # Catch-all for other unexpected errors
        print(f"An unexpected error occurred during Stage 1 separation in processor.py: {e}")
        import traceback

        traceback.print_exc()
        if temp_separation_subdir.exists():
            shutil.rmtree(temp_separation_subdir)
        return None


def run_stage2_enhancement(vocals_audio_path: Path, output_dir: Path, model_name: str) -> Path | None:
    """
    Performs Stage 2 speech enhancement on the separated vocal stem.
    Uses SpeechBrain for enhancement. Models are typically downloaded to a cache.

    Args:
        vocals_audio_path: Path to the (noisy) vocal stem from Stage 1.
        output_dir: The primary directory where final outputs should be stored.
                    A subdirectory for SpeechBrain models will be created here or in user cache.
        model_name: The SpeechBrain/Hugging Face model name for enhancement.

    Returns:
        Path to the enhanced vocal stem if successful, else None.
    """
    if SepformerEnhancement is None or torchaudio is None or Resample is None:
        print("Error: 'speechbrain' or 'torchaudio' not installed/found. Cannot perform enhancement.")
        return None

    print(f"\n--- Stage 2: Enhancing Dialogue using model '{model_name}' ---")
    print(f"Processing: {vocals_audio_path.name}")
    try:
        # Define a cache directory for SpeechBrain models. SpeechBrain handles actual caching.
        # This path helps organize where SpeechBrain might be instructed to save.
        cache_dir_enhancer = (
            output_dir / "pretrained_models_cache" / "speechbrain" / model_name.replace("/", "_")
        )
        cache_dir_enhancer.mkdir(parents=True, exist_ok=True)

        enhancer = SepformerEnhancement.from_hparams(
            source=model_name,
            savedir=str(cache_dir_enhancer),  # SpeechBrain uses this to store downloaded models
        )

        # Determine target sample rate based on common SpeechBrain model conventions.
        target_sr = 16000
        if "sepformer-dns4-16k" in model_name.lower():
            target_sr = 16000
        elif "sepformer-dns4-8k" in model_name.lower():
            target_sr = 8000

        noisy_speech, original_sr = torchaudio.load(vocals_audio_path)

        # Resample if the input vocal stem's sample rate differs from the model's target.
        if original_sr != target_sr:
            print(f"Resampling from {original_sr}Hz to {target_sr}Hz for enhancement model.")
            resampler = Resample(orig_freq=original_sr, new_freq=target_sr)
            noisy_speech = resampler(noisy_speech)

        # Ensure audio is mono and has the correct shape (Batch, Time) for SpeechBrain.
        if noisy_speech.ndim == 1:
            noisy_speech = noisy_speech.unsqueeze(0)  # Add batch dim
        if noisy_speech.shape[0] > 1:
            noisy_speech = torch.mean(noisy_speech, dim=0, keepdim=True)  # Convert to mono

        # Perform enhancement.
        enhanced_speech = enhancer.enhance_batch(noisy_speech, lengths=torch.tensor([noisy_speech.shape[1]]))

        # Save the enhanced audio.
        clean_input_stem = vocals_audio_path.stem.replace("_S1_vocals", "")  # Use a cleaner base name
        output_filename = output_dir / f"{clean_input_stem}_S2_enhanced.wav"
        torchaudio.save(output_filename, enhanced_speech.cpu().squeeze(0), target_sr)
        print(f"Enhanced vocals (Stage 2) saved to: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error during Stage 2 enhancement: {e}")
        import traceback

        traceback.print_exc()
        return None


def finalize_audio_output(
    input_path: Path, output_path: Path, target_sample_rate: int | None, convert_to_mono: bool
) -> Path | None:
    """
    Processes the final audio output: copies it, and optionally resamples and/or converts to mono.
    This ensures the final output adheres to specified formatting.

    Args:
        input_path: Path to the audio file to be finalized (e.g., from Stage 1 or Stage 2).
        output_path: Desired path for the final output file.
        target_sample_rate: If not None, resample to this rate (e.g., 16000).
        convert_to_mono: If True, convert to a mono audio track.

    Returns:
        Path to the finalized audio file, or the input_path if errors occur and copying fails.
    """
    if torchaudio is None or Resample is None:  # torchaudio.transforms.Resample
        print("Error: 'torchaudio' is required for resampling or mono conversion.")
        # Fallback: just copy if torchaudio is unavailable and paths differ
        if input_path != output_path:
            try:
                shutil.copy(input_path, output_path)
                print(f"Copied as-is (torchaudio unavailable) to: {output_path}")
                return output_path
            except Exception as copy_e:
                print(f"Error copying file during torchaudio fallback: {copy_e}")
                return input_path  # Return original if copy fails
        return input_path  # Return original if no processing and paths are same

    print(f"\n--- Finalizing Output Audio: {input_path.name} to {output_path.name} ---")
    try:
        if not input_path.exists():
            print(f"Error: Input file for finalization not found: {input_path}")
            return None

        # If no processing is needed (no resampling, no mono conversion)
        # and the paths are different, just copy the file.
        if input_path != output_path and not target_sample_rate and not convert_to_mono:
            shutil.copy(input_path, output_path)
            print(f"Final output (copied as-is): {output_path}")
            return output_path
        # If paths are the same and no processing is needed, do nothing.
        elif input_path == output_path and not target_sample_rate and not convert_to_mono:
            print(f"Final output (no changes needed, paths are same): {output_path}")
            return output_path

        # Load waveform for processing if resampling or mono conversion is needed.
        waveform, sr = torchaudio.load(input_path)
        did_process = False  # Flag to track if any modification was made

        if convert_to_mono and waveform.shape[0] > 1:  # waveform.shape[0] is number of channels
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print("Converted to mono.")
            did_process = True

        if target_sample_rate and sr != target_sample_rate:
            print(f"Resampling from {sr}Hz to {target_sample_rate}Hz...")
            resampler = Resample(orig_freq=sr, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sr = target_sample_rate  # Update sample rate to the new one for saving
            print(f"Resampled to {sr}Hz.")
            did_process = True

        # Save the processed file if any processing occurred or if output path is different.
        if did_process or input_path != output_path:
            torchaudio.save(output_path, waveform, sr, format="wav")  # Explicitly save as WAV
            print(f"Final processed audio saved to: {output_path}")
        else:  # Paths are the same, and no processing was done.
            print(f"Final audio (no changes made, paths are same): {output_path}")

        return output_path

    except Exception as e:
        print(f"Error during audio finalization: {e}")
        import traceback

        traceback.print_exc()
        # Fallback: attempt to copy original if processing fails and output_path is different
        if input_path != output_path:
            try:
                shutil.copy(input_path, output_path)
                print(f"Warning: Finalization failed. Copied original file to: {output_path}")
                return output_path
            except Exception as copy_e:
                print(f"Error copying original file during fallback: {copy_e}")
        return input_path  # Return original if in-place processing failed or copy failed
