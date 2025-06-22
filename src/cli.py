#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm

from sepaudio_lib.utils import is_url
from sepaudio_lib.downloader import download_audio_from_url, extract_audio_from_local_video
from sepaudio_lib.processor import run_stage1_separation, run_stage2_enhancement, finalize_audio_output

def main():
    parser = argparse.ArgumentParser(
        description="Sepaudio: Downloads, separates, and enhances dialogue from audio/video.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_source",
        help="Path to a local audio/video file OR a URL (e.g., YouTube video)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="processed_dialogue_output",
        help="Directory to save all output files (default: processed_dialogue_output)."
    )
    parser.add_argument(
        "--separator_model_name",
        default="UVR-MDX-NET-Voc_FT",
        help="Model name for Stage 1 audio separation (from 'audio-separator' library)."
    )
    parser.add_argument(
        "--enable_enhancement",
        action="store_true",
        help="Enable Stage 2 speech enhancement on the separated vocals."
    )
    parser.add_argument(
        "--enhancer_model_name",
        default="speechbrain/sepformer-dns4-16k-enhancement",
        help="Model name for Stage 2 speech enhancement."
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Explicitly treat a local input file as video."
    )
    parser.add_argument(
        "--keep_intermediate_files",
        action="store_true",
        help="Keep all intermediate files."
    )
    parser.add_argument(
        "--output_sample_rate",
        type=int,
        default=None,
        help="Resample final dialogue to this sample rate (e.g., 16000)."
    )
    parser.add_argument(
        "--output_mono",
        action="store_true",
        help="Convert final dialogue output to mono."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints."
    )

    args = parser.parse_args()

    if args.debug:
        print(f"DEBUG: args.input_source value: '{args.input_source}'")
        print(f"DEBUG: type of args.input_source: {type(args.input_source)}")
        print(f"DEBUG: is_url(args.input_source) returns: {is_url(args.input_source)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_audio_file: Path | None = None
    temp_files_to_cleanup = []

    num_main_stages = 1
    if args.separator_model_name: num_main_stages += 1
    if args.enable_enhancement: num_main_stages += 1
    num_main_stages +=1

    with tqdm(total=num_main_stages, desc="Overall Progress", unit="stage", dynamic_ncols=True) as pbar_main:
        pbar_main.set_description("Stage: Input/Download")
        if is_url(args.input_source):
            temp_dl_dir = output_dir / "temp_download" # More specific temp dir
            temp_dl_dir.mkdir(parents=True, exist_ok=True)
            current_audio_file = download_audio_from_url(args.input_source, temp_dl_dir)
            if current_audio_file and not args.keep_intermediate_files:
                temp_files_to_cleanup.append(current_audio_file)
        else:
            local_file_path = Path(args.input_source)
            if not local_file_path.exists():
                print(f"Error: Local file not found: {local_file_path}")
                sys.exit(1)
            video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']
            is_video_input = args.video or local_file_path.suffix.lower() in video_extensions
            if is_video_input:
                temp_extract_dir = output_dir / "temp_extracted_audio" # More specific
                temp_extract_dir.mkdir(parents=True, exist_ok=True)
                current_audio_file = extract_audio_from_local_video(local_file_path, temp_extract_dir)
                if current_audio_file and not args.keep_intermediate_files:
                    temp_files_to_cleanup.append(current_audio_file)
            else:
                current_audio_file_name = f"{local_file_path.stem}_inputcopy{local_file_path.suffix}"
                current_audio_file = output_dir / "temp_input_copies" / current_audio_file_name # Store copies in a temp subdir
                current_audio_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(local_file_path, current_audio_file)
                print(f"Copied local audio to: {current_audio_file}")
                if not args.keep_intermediate_files:
                     temp_files_to_cleanup.append(current_audio_file)
        pbar_main.update(1)

        if not current_audio_file or not current_audio_file.exists():
            print("Error: Could not obtain a valid audio file to process.")
            sys.exit(1)

        pbar_main.set_description("Stage: Separation")
        stage1_vocals_path = run_stage1_separation(current_audio_file, output_dir, args.separator_model_name)
        pbar_main.update(1)

        if not stage1_vocals_path:
            print("Stage 1 separation failed. Exiting.")
            sys.exit(1)
        
        processed_audio_path_before_finalize = stage1_vocals_path
        if (args.enable_enhancement or args.output_sample_rate or args.output_mono) and not args.keep_intermediate_files:
            if stage1_vocals_path not in temp_files_to_cleanup:
                temp_files_to_cleanup.append(stage1_vocals_path)

        if args.enable_enhancement:
            pbar_main.set_description("Stage: Enhancement")
            stage2_enhanced_path = run_stage2_enhancement(stage1_vocals_path, output_dir, args.enhancer_model_name)
            pbar_main.update(1)
            if stage2_enhanced_path:
                processed_audio_path_before_finalize = stage2_enhanced_path
                if not args.keep_intermediate_files and stage1_vocals_path not in temp_files_to_cleanup:
                     temp_files_to_cleanup.append(stage1_vocals_path)
            else:
                print("Stage 2 enhancement failed. Using Stage 1 output for finalization.")
                if stage1_vocals_path in temp_files_to_cleanup and not (args.output_sample_rate or args.output_mono):
                     if stage1_vocals_path in temp_files_to_cleanup: temp_files_to_cleanup.remove(stage1_vocals_path)

        pbar_main.set_description("Stage: Finalization")
        final_base_name = Path(processed_audio_path_before_finalize.stem.replace("_S1_vocals", "").replace("_S2_enhanced", "").replace("_inputcopy",""))
        final_output_filename = output_dir / f"{final_base_name}_dialogue.wav"
        final_processed_path = finalize_audio_output(processed_audio_path_before_finalize, final_output_filename, args.output_sample_rate, args.output_mono)
        pbar_main.update(1)

        if final_processed_path != processed_audio_path_before_finalize and \
           processed_audio_path_before_finalize not in temp_files_to_cleanup and \
           not args.keep_intermediate_files:
            temp_files_to_cleanup.append(processed_audio_path_before_finalize)

    print("\n--- Processing Complete ---")
    if final_processed_path and final_processed_path.exists():
        print(f"Final processed dialogue track: {final_processed_path.resolve()}")
    else:
        print("No final output was successfully generated.")

    if not args.keep_intermediate_files:
        print("\nCleaning up intermediate files...")
        for f_path in set(temp_files_to_cleanup):
            if f_path and f_path.exists() and (not final_processed_path or f_path.resolve() != final_processed_path.resolve()):
                try:
                    if f_path.is_dir(): # For temp download/extract dirs
                        shutil.rmtree(f_path)
                        print(f"  - Removed directory: {f_path.name}")
                    else:
                        f_path.unlink()
                        print(f"  - Removed file: {f_path.name}")
                except OSError as e:
                    print(f"Warning: Could not remove {f_path.name}: {e}")
        # Clean up potentially empty main temp directories if they were created
        for temp_main_dir_name in ["temp_download", "temp_extracted_audio", "temp_input_copies"]:
            temp_main_dir = output_dir / temp_main_dir_name
            if temp_main_dir.exists() and not any(temp_main_dir.iterdir()):
                try:
                    shutil.rmtree(temp_main_dir)
                    print(f"  - Removed empty main temp directory: {temp_main_dir.name}")
                except OSError as e:
                    print(f"Warning: Could not remove empty main temp directory {temp_main_dir.name}: {e}")
    else:
        print(f"\nIntermediate files kept in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()