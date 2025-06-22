# test_sep.py
from audio_separator.separator import Separator
import os
import shutil
import traceback # Import traceback module
import logging   # <--- IMPORT LOGGING MODULE

output_dir = 'test_sep_output'
# Clean up previous test output if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

dummy_audio_path = "test.wav" 
if not os.path.exists(dummy_audio_path):
    print(f"ERROR: Test audio file '{dummy_audio_path}' does not exist. Please create it or use a valid path.")
    # Example: Create a silent WAV if soundfile is available
    try:
        import soundfile as sf
        import numpy as np
        samplerate = 44100; duration = 1; data = np.zeros(int(samplerate * duration))
        sf.write(dummy_audio_path, data, samplerate)
        print(f"Created dummy audio file: {dummy_audio_path}")
    except ImportError:
        print("Please create woman.wav manually or install soundfile (pip install soundfile numpy) to auto-create it for testing.")
        exit()


print(f"Attempting to process dummy file: {dummy_audio_path}")

try:
    separator = Separator(
        output_dir=output_dir,
        log_level=logging.INFO  # <--- USE logging.INFO (the integer constant)
    )
    
    print(f"Loading model UVR-MDX-NET-Voc_FT...")
    # This will try to download the model if not found in the default cache
    separator.load_model('UVR-MDX-NET-Voc_FT') 
    print("Model loaded successfully.")
    
    print(f"Separating audio file: {dummy_audio_path}")
    output_files = separator.separate(dummy_audio_path)
    print(f"Separation complete. Output files: {output_files}")

except TypeError as te:
    print(f"Test TypeError: {te}")
    print("--- Full Traceback for TypeError ---")
    traceback.print_exc() 
    print("------------------------------------")
except Exception as e:
    print(f"Test Exception: {e}")
    print("--- Full Traceback for Exception ---")
    traceback.print_exc()
    print("------------------------------------")