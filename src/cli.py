#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import logging 

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
        print(f"DEBUG: Input source: '{args.input_source}'")
        print(f"DEBUG: is_url() returns: {is_url(args.input_source)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_audio_file: Path | None = None
    temp_files_to_cleanup = [] # Stores paths of files this script creates that might be intermediate

    # Determine number of main stages for tqdm
    num_main_stages = 1 # Input/Download
    if args.separator_model_name: num_main_stages += 1
    if args.enable_enhancement: num_main_stages += 1
    num_main_stages +=1 # Finalization/Copy

    with tqdm(total=num_main_stages, desc="Overall Progress", unit="stage", dynamic_ncols=True) as pbar_main:
        pbar_main.set_description("Stage: Input/Download")
        if is_url(args.input_source):
            # Create a specific subdirectory for downloaded files to make cleanup easier
            temp_dl_dir = output_dir / "temp_downloaded_audio"
            temp_dl_dir.mkdir(parents=True, exist_ok=True)
            current_audio_file = download_audio_from_url(args.input_source, temp_dl_dir)
            # The downloaded file itself is intermediate if we don't keep intermediates
            if current_audio_file and not args.keep_intermediate_files:
                temp_files_to_cleanup.append(current_audio_file)
        else: # Local file processing
            local_file_path = Path(args.input_source)
            if not local_file_path.exists():
                print(f"Error: Local file not found: {local_file_path}")
                sys.exit(1)
            
            video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']
            is_video_input = args.video or local_file_path.suffix.lower() in video_extensions

            if is_video_input:
                temp_extract_dir = output_dir / "temp_extracted_audio"
                temp_extract_dir.mkdir(parents=True, exist_ok=True)
                current_audio_file = extract_audio_from_local_video(local_file_path, temp_extract_dir)
                if current_audio_file and not args.keep_intermediate_files:
                    temp_files_to_cleanup.append(current_audio_file)
            else: # Local audio file
                # Copy to a working directory to avoid modifying original and for consistent cleanup
                temp_input_copy_dir = output_dir / "temp_input_copies"
                temp_input_copy_dir.mkdir(parents=True, exist_ok=True)
                current_audio_file_name = f"{local_file_path.stem}_inputcopy{local_file_path.suffix}"
                current_audio_file = temp_input_copy_dir / current_audio_file_name
                shutil.copy(local_file_path, current_audio_file)
                print(f"Copied local audio to temporary location: {current_audio_file}")
                if not args.keep_intermediate_files:
                     temp_files_to_cleanup.append(current_audio_file)
        pbar_main.update(1)


        if not current_audio_file or not current_audio_file.exists():
            print("Error: Could not obtain a valid audio file to process.")
            # Clean up any temp dirs created before exiting
            for temp_dir_name in ["temp_downloaded_audio", "temp_extracted_audio", "temp_input_copies"]:
                temp_dir_path = output_dir / temp_dir_name
                if temp_dir_path.exists():
                    shutil.rmtree(temp_dir_path)
            sys.exit(1)

        pbar_main.set_description("Stage: Separation")
        stage1_vocals_path = run_stage1_separation(current_audio_file, output_dir, args.separator_model_name)
        pbar_main.update(1)

        if not stage1_vocals_path:
            print("Stage 1 separation failed. Exiting.")
            sys.exit(1) # Or handle more gracefully, cleanup temp files before exit
        
        processed_audio_path_before_finalize = stage1_vocals_path
        # If S1 output will be further processed (enhanced or reformatted) AND we are not keeping intermediates,
        # then S1 output is temporary.
        if (args.enable_enhancement or args.output_sample_rate or args.output_mono) and \
           not args.keep_intermediate_files and \
           stage1_vocals_path not in temp_files_to_cleanup:
            temp_files_to_cleanup.append(stage1_vocals_path)

        if args.enable_enhancement:
            pbar_main.set_description("Stage: Enhancement")
            stage2_enhanced_path = run_stage2_enhancement(stage1_vocals_path, output_dir, args.enhancer_model_name)
            
            if stage2_enhanced_path:
                processed_audio_path_before_finalize = stage2_enhanced_path
                # S1 is definitely intermediate if S2 succeeds and we don't keep intermediates.
                if not args.keep_intermediate_files and stage1_vocals_path not in temp_files_to_cleanup:
                     temp_files_to_cleanup.append(stage1_vocals_path)
                # S2 output is intermediate if it will be further processed by finalize AND we don't keep intermediates
                if (args.output_sample_rate or args.output_mono) and \
                   not args.keep_intermediate_files and \
                   stage2_enhanced_path not in temp_files_to_cleanup:
                    temp_files_to_cleanup.append(stage2_enhanced_path)
            else: # Enhancement failed
                print("Stage 2 enhancement failed. Using Stage 1 output for finalization.")
                # If S1 was marked as temp because enhancement was enabled, but enhancement failed,
                # S1 might become the input to finalization. If finalization makes no changes, it's not temp.
                if stage1_vocals_path in temp_files_to_cleanup and not (args.output_sample_rate or args.output_mono):
                     temp_files_to_cleanup.remove(stage1_vocals_path)
            pbar_main.update(1) # Update pbar even if enhancement failed, as the stage was attempted
        
        pbar_main.set_description("Stage: Finalization")
        # Ensure a clean base name for the final output by removing stage-specific suffixes
        clean_base_name = Path(current_audio_file.stem.replace('_extracted', '').replace('_inputcopy', ''))
        final_output_filename = output_dir / f"{clean_base_name}_dialogue.wav"
        
        final_processed_path = finalize_audio_output(
            processed_audio_path_before_finalize,
            final_output_filename,
            args.output_sample_rate,
            args.output_mono
        )
        pbar_main.update(1)

        # If finalization created a new file, and the pre-finalized one is different,
        # the pre-finalized one is temp (if not already marked and not keeping intermediates).
        if final_processed_path and final_processed_path != processed_audio_path_before_finalize and \
           processed_audio_path_before_finalize not in temp_files_to_cleanup and \
           not args.keep_intermediate_files:
            temp_files_to_cleanup.append(processed_audio_path_before_finalize)

    # --- End of TQDM progress bar ---

    print("\n--- Processing Complete ---")
    if final_processed_path and final_processed_path.exists():
        print(f"Final processed dialogue track: {final_processed_path.resolve()}")
    else:
        print("No final output was successfully generated or found.")

    # --- Cleanup ---
    if not args.keep_intermediate_files:
        print("\nCleaning up intermediate files...")
        # Cleanup individual files
        for f_path in set(temp_files_to_cleanup): # Use set to avoid duplicates
            # Ensure we don't delete the final product if it was an in-place modification
            if f_path and f_path.exists() and (not final_processed_path or f_path.resolve() != final_processed_path.resolve()):
                try:
                    if f_path.is_file():
                        f_path.unlink()
                        print(f"  - Removed file: {f_path.name}")
                    # Directories handled separately below to ensure they are empty or specifically marked for cleanup
                except OSError as e:
                    print(f"Warning: Could not remove temp file {f_path.name}: {e}")
        
        # Cleanup temporary directories if they are empty or fully managed
        for temp_dir_name in ["temp_downloaded_audio", "temp_extracted_audio", "temp_input_copies"]:
            temp_dir_path = output_dir / temp_dir_name
            if temp_dir_path.exists():
                try:
                    # Check if empty before removing, or if it was a fully temp dir
                    # For simplicity, if not keeping intermediates, we assume these top-level temp dirs can be removed.
                    shutil.rmtree(temp_dir_path)
                    print(f"  - Removed temporary directory: {temp_dir_path.name}")
                except OSError as e:
                    print(f"Warning: Could not remove temporary directory {temp_dir_path.name}: {e}")
    else:
        print(f"\nIntermediate files and directories kept in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()