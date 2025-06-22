import os
import shutil
import logging
import traceback
from pathlib import Path

try:
    from audio_separator.separator import Separator
except ImportError:
    print("ERROR: audio-separator library not found. Please install it.")
    Separator = None

try:
    import soundfile as sf
    import numpy as np
except ImportError:
    sf = None
    np = None
    print("WARNING: soundfile or numpy not found. Cannot auto-create dummy audio.")

def create_dummy_audio(file_path: Path, duration_sec: int = 2, sample_rate: int = 44100) -> bool:
    """Creates a simple non-silent WAV file if soundfile and numpy are available."""
    if sf and np:
        if not file_path.exists():
            print(f"Creating dummy audio file: {file_path} ({duration_sec}s, {sample_rate}Hz)")
            try:
                frequency = 440  # A4 note
                t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
                # Create a simple sine wave, ensure it's float32 for common WAV format
                data = (0.1 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
                # Add a little bit of noise to make it less "empty"
                noise = (np.random.rand(len(data)) - 0.5) * 0.01
                data += noise.astype(np.float32)
                data = np.clip(data, -1.0, 1.0) # Ensure data is within valid range for float32 WAV

                sf.write(file_path, data, sample_rate, subtype='FLOAT') # Save as float32
                print(f"Successfully created dummy audio: {file_path}")
                return True
            except Exception as e:
                print(f"Error creating dummy audio file {file_path}: {e}")
                return False
        # print(f"Dummy audio file {file_path} already exists.") # Less verbose
        return True
    else:
        if not file_path.exists():
            print(f"Cannot auto-create {file_path} because soundfile/numpy are missing and file does not exist.")
        return file_path.exists()

def run_separator_test(model_to_load_filename: str, audio_file_path_str: str, test_output_dir_str: str):
    if Separator is None:
        print("Separator class not available. Skipping test.")
        return

    print(f"\n--- Testing Separator with model filename: '{model_to_load_filename}' ---")
    audio_file_path = Path(audio_file_path_str)
    model_output_subdir_name = Path(model_to_load_filename).stem
    model_specific_output_dir = Path(test_output_dir_str) / model_output_subdir_name
    
    if model_specific_output_dir.exists(): shutil.rmtree(model_specific_output_dir)
    model_specific_output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_file_path.exists():
        print(f"ERROR: Test audio file '{audio_file_path}' does not exist. Skipping test for '{model_to_load_filename}'.")
        return

    try:
        print("DEBUG: Initializing Separator with: output_dir, log_level=logging.INFO (OMITTING model_file_dir COMPLETELY)")
        separator = Separator(
            output_dir=str(model_specific_output_dir), 
            log_level=logging.INFO
        )
        
        actual_model_dir_used = "Unknown"
        if hasattr(separator, 'model_file_dir') and separator.model_file_dir is not None:
            actual_model_dir_used = separator.model_file_dir
        print(f"DEBUG: Separator instance is configured to use model directory: {actual_model_dir_used}")

        print(f"Loading model using filename '{model_to_load_filename}'...")
        separator.load_model(model_to_load_filename) 
        print(f"Model '{model_to_load_filename}' loaded successfully.")
        
        print(f"Separating audio file: {audio_file_path.name}...")
        output_files = separator.separate(str(audio_file_path))
        
        if output_files:
            print(f"Separation complete for '{model_to_load_filename}'. Output files:")
            for f_str in output_files: print(f"  - {Path(f_str).name}")
        else:
            print(f"Separation for '{model_to_load_filename}' produced no output files.")

    except ValueError as ve:
        print(f"TEST SCRIPT ValueError for model '{model_to_load_filename}': {ve}")
        print("--- Full Traceback for ValueError ---"); traceback.print_exc(); print("---")
    except TypeError as te:
        print(f"TEST SCRIPT TypeError for model '{model_to_load_filename}': {te}")
        print("--- Full Traceback for TypeError ---"); traceback.print_exc(); print("---")
    except Exception as e:
        print(f"TEST SCRIPT Generic Exception for model '{model_to_load_filename}': {e}")
        print("--- Full Traceback for Generic Exception ---"); traceback.print_exc(); print("---")

if __name__ == "__main__":
    print("Running audio-separator direct test script...")
    print("This test OMITS `model_file_dir` from Separator() constructor call, and uses FULL model filenames.")

    dummy_audio_filename = "test_audio.wav" 
    dummy_audio_file_path = Path(dummy_audio_filename)
    
    if not create_dummy_audio(dummy_audio_file_path):
        if not dummy_audio_file_path.exists():
            print(f"FATAL: Could not create or find '{dummy_audio_filename}'. Aborting test.")
            exit()
        else:
            print(f"Using existing test audio file: '{dummy_audio_filename}'.")

    main_test_output_folder = Path("separator_test_outputs")

    print("\n" + "="*30 + " TESTING UVR-MDX-NET-Voc_FT.onnx " + "="*30)
    run_separator_test(
        model_to_load_filename='UVR-MDX-NET-Voc_FT.onnx', # Pass the full filename
        audio_file_path_str=str(dummy_audio_file_path),
        test_output_dir_str=str(main_test_output_folder)
    )

    print("\n" + "="*30 + " TESTING htdemucs_ft.yaml " + "="*30)
    run_separator_test(
        model_to_load_filename='htdemucs_ft.yaml', # Pass the full filename
        audio_file_path_str=str(dummy_audio_file_path),
        test_output_dir_str=str(main_test_output_folder)
    )
    
    print("\n--- Test script finished ---")