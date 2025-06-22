from audio_separator.separator import Separator
try:
    # Test 1: Common API
    separator = Separator(output_dir='test_sep_output')
    separator.load_model('UVR-MDX-NET-Voc_FT') # or any model you use
    print("Separator initialized and model loaded successfully (common API).")

except TypeError as te:
    print(f"Test TypeError: {te}")
except Exception as e:
    print(f"Test Exception: {e}")