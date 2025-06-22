# test_sep.py (Version to ensure model_file_dir is not set)
import os
import shutil
import logging
import traceback
from pathlib import Path

try:
    from audio_separator.separator import Separator
    # Attempt to see Separator's default config path if an attribute exists
    # This is speculative, depends on library internals
    try:
        DEFAULT_MODEL_REPO = Separator.DEFAULT_MODEL_REPO
        DEFAULT_CACHE_DIR = Separator.DEFAULT_CACHE_DIR
        print(f"DEBUG: audio_separator.Separator default repo: {DEFAULT_MODEL_REPO}")
        print(f"DEBUG: audio_separator.Separator default cache dir: {DEFAULT_CACHE_DIR}")
    except AttributeError:
        print("DEBUG: audio_separator.Separator does not expose DEFAULT_MODEL_REPO or DEFAULT_CACHE_DIR directly.")

except ImportError:
    print("ERROR: audio-separator library not found.")
    Separator = None

try:
    import soundfile as sf
    import numpy as np
except ImportError:
    sf = None
    np = None
    print("WARNING: soundfile or numpy not found. Cannot auto-create dummy audio.")

def create_dummy_audio(file_path: Path, duration_sec: int = 1, sample_rate: int = 44100):
    if sf and np:
        if not file_path.exists():
            print(f"Creating dummy audio file: {file_path} ({duration_sec}s, {sample_rate}Hz)")
            data = np.zeros(int(sample_rate * duration_sec))
            sf.write(file_path, data, sample_rate)
            return True
        return True
    return False

def run_separator_test(model_to_test: str, audio_file_path_str: str, test_output_dir_str: str):
    if Separator is None: return
    print(f"\n--- Testing Separator with model: '{model_to_test}' ---")
    audio_file_path = Path(audio_file_path_str); test_output_dir = Path(test_output_dir_str)
    model_specific_output_dir = test_output_dir / model_to_test
    if model_specific_output_dir.exists(): shutil.rmtree(model_specific_output_dir)
    model_specific_output_dir.mkdir(parents=True, exist_ok=True)
    if not audio_file_path.exists():
        print(f"ERROR: Test audio file '{audio_file_path}' does not exist. Skipping."); return

    try:
        print("DEBUG: Initializing Separator with: output_dir, log_level=logging.INFO")
        separator = Separator(
            output_dir=str(model_specific_output_dir),
            log_level=logging.INFO
            # Explicitly NO model_file_dir
        )
        # Check if model_file_dir attribute exists and what its value is AFTER initialization
        if hasattr(separator, 'model_file_dir'):
            print(f"DEBUG: separator.model_file_dir AFTER init: {separator.model_file_dir}")
        else:
            print("DEBUG: separator object has no 'model_file_dir' attribute directly exposed after init.")


        print(f"Loading model '{model_to_test}'...")
        separator.load_model(model_to_test) 
        print(f"Model '{model_to_test}' loaded successfully.")
        print(f"Separating audio file: {audio_file_path.name}")
        output_files = separator.separate(str(audio_file_path))
        if output_files: print(f"Separation complete for '{model_to_test}'. Outputs: {[Path(f).name for f in output_files]}")
        else: print(f"Separation for '{model_to_test}' produced no output files.")
    except ValueError as ve:
        print(f"TEST SCRIPT ValueError for model '{model_to_test}': {ve}")
        print("--- Full Traceback for ValueError ---"); traceback.print_exc(); print("---")
    except TypeError as te:
        print(f"TEST SCRIPT TypeError for model '{model_to_test}': {te}")
        print("--- Full Traceback for TypeError ---"); traceback.print_exc(); print("---")
    except Exception as e:
        print(f"TEST SCRIPT Generic Exception for model '{model_to_test}': {e}")
        print("--- Full Traceback for Exception ---"); traceback.print_exc(); print("---")

if __name__ == "__main__":
    print("Running audio-separator direct test script (double check Separator init)...")
    dummy_audio_filename = "test_audio.wav"
    dummy_audio_file_path = Path(dummy_audio_filename)
    if not create_dummy_audio(dummy_audio_file_path):
        if not dummy_audio_file_path.exists(): print(f"FATAL: Could not create or find '{dummy_audio_filename}'. Aborting."); exit()
        else: print(f"Using existing '{dummy_audio_filename}'.")
    main_test_output_folder = Path("separator_test_outputs")
    
    run_separator_test('UVR-MDX-NET-Voc_FT', str(dummy_audio_file_path), str(main_test_output_folder))
    run_separator_test('htdemucs_ft', str(dummy_audio_file_path), str(main_test_output_folder))
    
    print("\n--- Test script finished ---")