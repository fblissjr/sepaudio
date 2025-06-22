import os
import shutil
import logging # For logging.INFO
import traceback
from pathlib import Path

# Attempt to import Separator, handle if not installed
try:
    from audio_separator.separator import Separator
except ImportError:
    print("ERROR: audio-separator library not found. Please install it.")
    print("Try: pip install audio-separator[gpu] (or without [gpu] for CPU)")
    Separator = None # So the script doesn't crash immediately if it's not there

# Attempt to import soundfile for creating a dummy audio file
try:
    import soundfile as sf
    import numpy as np
except ImportError:
    sf = None
    np = None
    print("WARNING: soundfile or numpy not found. Cannot auto-create dummy audio.")
    print("Please create 'test_audio.wav' manually for this test script to run properly.")

def create_dummy_audio(file_path: Path, duration_sec: int = 1, sample_rate: int = 44100):
    """Creates a simple silent WAV file if soundfile and numpy are available."""
    if sf and np:
        if not file_path.exists():
            print(f"Creating dummy audio file: {file_path} ({duration_sec}s, {sample_rate}Hz)")
            data = np.zeros(int(sample_rate * duration_sec))
            sf.write(file_path, data, sample_rate)
            return True
        return True # File already exists
    return False # Cannot create

def run_separator_test(model_to_test: str, audio_file_path_str: str, test_output_dir_str: str):
    """
    Tests a specific model from the audio-separator library.
    """
    if Separator is None:
        return

    print(f"\n--- Testing Separator with model: '{model_to_test}' ---")
    print(f"--- Using audio file: '{audio_file_path_str}' ---")
    print(f"--- Output will be in: '{test_output_dir_str}' ---")

    audio_file_path = Path(audio_file_path_str)
    test_output_dir = Path(test_output_dir_str)

    # Clean up previous test output for this model if it exists
    model_specific_output_dir = test_output_dir / model_to_test
    if model_specific_output_dir.exists():
        shutil.rmtree(model_specific_output_dir)
    model_specific_output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_file_path.exists():
        print(f"ERROR: Test audio file '{audio_file_path}' does not exist. Skipping test for this model.")
        return

    try:
        # Initialize Separator:
        # - output_dir: Where this specific Separator instance will save its stems.
        # - log_level: Use logging.INFO (integer constant).
        # - IMPORTANT: DO NOT set model_file_dir. Let audio-separator use its default
        #   cache directory so it can download models.
        separator = Separator(
            output_dir=str(model_specific_output_dir),
            log_level=logging.INFO
        )
        
        print(f"Loading model '{model_to_test}'...")
        separator.load_model(model_to_test) 
        print(f"Model '{model_to_test}' loaded successfully.")
        
        print(f"Separating audio file: {audio_file_path.name}")
        output_files = separator.separate(str(audio_file_path))
        
        if output_files:
            print(f"Separation complete for model '{model_to_test}'. Output files:")
            for f in output_files:
                print(f"  - {Path(f).name}")
        else:
            print(f"Separation for model '{model_to_test}' produced no output files.")

    except TypeError as te:
        print(f"TEST SCRIPT TypeError for model '{model_to_test}': {te}")
        print("This usually means an issue with how Separator or its methods are called, or an internal type mismatch.")
        print("--- Full Traceback for TypeError ---")
        traceback.print_exc() 
        print("------------------------------------")
    except ValueError as ve: # Catching specific ValueError for model not found
        print(f"TEST SCRIPT ValueError for model '{model_to_test}': {ve}")
        print("This often means the model name is incorrect or the model files could not be downloaded/found.")
        print("--- Full Traceback for ValueError ---")
        traceback.print_exc()
        print("------------------------------------")
    except Exception as e:
        print(f"TEST SCRIPT Generic Exception for model '{model_to_test}': {e}")
        print("--- Full Traceback for Exception ---")
        traceback.print_exc()
        print("------------------------------------")

if __name__ == "__main__":
    print("Running audio-separator direct test script...")

    # Define the dummy audio file (it will be created if it doesn't exist and soundfile is available)
    dummy_audio_filename = "test_audio.wav"
    dummy_audio_file_path = Path(dummy_audio_filename)
    
    # Create the dummy audio file for the test
    if not create_dummy_audio(dummy_audio_file_path):
        if not dummy_audio_file_path.exists():
            print(f"FATAL: Could not create or find '{dummy_audio_filename}'. Aborting test.")
            exit()
        else:
            print(f"Using existing '{dummy_audio_filename}'.")


    # Define a general output directory for all test outputs
    main_test_output_folder = Path("separator_test_outputs")

    # --- Test Case 1: UVR-MDX-NET-Voc_FT ---
    # This is the model that was causing issues before
    run_separator_test(
        model_to_test='UVR-MDX-NET-Voc_FT',
        audio_file_path_str=str(dummy_audio_file_path),
        test_output_dir_str=str(main_test_output_folder / "uvr_mdx_net_test")
    )

    # --- Test Case 2: htdemucs_ft (another common model) ---
    run_separator_test(
        model_to_test='htdemucs_ft',
        audio_file_path_str=str(dummy_audio_file_path),
        test_output_dir_str=str(main_test_output_folder / "htdemucs_ft_test")
    )
    
    # --- Test Case 3: A potentially simpler MDX model (if one exists and is small) ---
    # Example: 'UVR_MDXNET_Main.onnx' (check your list_models output for a suitable one)
    # run_separator_test(
    #     model_to_test='UVR_MDXNET_Main', # Just an example, use a valid name
    #     audio_file_path_str=str(dummy_audio_file_path),
    #     test_output_dir_str=str(main_test_output_folder / "uvr_mdxnet_main_test")
    # )

    print("\n--- Test script finished ---")
    print(f"Please check the '{main_test_output_folder}' directory for separation results.")
    if dummy_audio_file_path.exists() and "dummy" in dummy_audio_file_path.name : # Optional: Clean up dummy audio
         # dummy_audio_file_path.unlink()
         # print(f"Cleaned up dummy audio file: {dummy_audio_filename}")
         pass