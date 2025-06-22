## sepaudio

- download audio directly from http urls
- process local audio and video files
- two-stage approach to separate audio/dialogue, then reduce noise and improve clarity
- configurable model selection for both stages

### examples

```bash
sepaudio "https://www.youtube.com/watch?v=your_video_id" \
       --output_dir ./output \
       --separator_model_name "htdemucs_ft" \
       --enable_enhancement \
       --enhancer_model_name "speechbrain/sepformer-dns4-16k-enhancement" \
       --keep_intermediate_files
```

```bash
sepaudio audio.mp3 \
       --output_dir ./local_audio_processed \
       --separator_model_name "UVR-MDX-NET-Voc_FT"
```
