## sepaudio

- download audio directly from http urls
- process local audio and video files
- two-stage approach to separate audio/dialogue, then reduce noise and improve clarity
- configurable model selection for both stages

### usage examples

**2. from a http url:**

```bash
sepaudio "https://geocities.com/myvideos" \
       --output_dir ./my_output \
       --separator_model_name "htdemucs_ft" \
       --enable_enhancement \
       --enhancer_model_name "speechbrain/sepformer-dns4-16k-enhancement" \
       --keep_intermediate_files \
       --output_sample_rate 16000 \
       --output_mono
```

**2. from a local audio file, separation only, outputting at 44.1kHz stereo:**

```bash
sepaudio your_audio.mp3 \
       --output_dir ./local_audio_processed \
       --separator_model_name "UVR-MDX-NET-Voc_FT"
```

*(To change output sample rate or convert to mono, add `--output_sample_rate HERTZ` and/or `--output_mono` respectively.)*

**3. from a local video file, separation only:**

```bash
sepaudio /path/to/your_video.mp4 \
    --output_dir ./processed_video_audio
```

### key cli arguments

- `input_source`: Path to a local audio/video file OR a URL.
- `--output_dir`: Directory to save all output files (default: `processed_dialogue_output`).
- `--separator_model_name`: Model for Stage 1 separation (e.g., `UVR-MDX-NET-Voc_FT`, `htdemucs_ft`). Default is `UVR-MDX-NET-Voc_FT`.
- `--enable_enhancement`: Flag to enable Stage 2 speech enhancement.
- `--enhancer_model_name`: Model for Stage 2 enhancement (e.g., `speechbrain/sepformer-dns4-16k-enhancement`).
- `--keep_intermediate_files`: Flag to keep all intermediate files.
- `--output_sample_rate HERTZ`: Resample the final dialogue output to this sample rate (e.g., 16000). Requires `torchaudio`.
- `--output_mono`: Convert the final dialogue output to mono. Requires `torchaudio`.
- `--video`: Explicitly treat a local input file as video if its extension is ambiguous.
