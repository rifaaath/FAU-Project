import subprocess
import sys
from pathlib import Path
import argparse  # Import argparse to handle the --resume flag

EXPERIMENT_NAME = "yolo_medium_final_run_v4"

YOLO_WEIGHTS_PATH = "yolo_runs_clean_medium/train/weights/best.pt"
STYLE_ENCODER_PATH = "checkpoints/supcon_encoder.pt"
CLEAN_IMAGES_DIR = "datasets/HomerComp_Cleaned/"
METADATA_JSON_PATH = "papytwin/HomerCompTraining/HomerCompTraining.json"
METADATA_EXCEL_PATH = "papytwin/1.CompetitionOverview.xlsx"
ORIGINAL_TRAIN_SPLIT = "final_splits/train_final.csv"
ORIGINAL_TEST_SPLIT = "final_splits/test_final.csv"

OUTPUT_DIR = Path(f"pipeline_runs/{EXPERIMENT_NAME}")
PREDICTIONS_JSON = OUTPUT_DIR / "1_predictions/predict/predictions_by_filename.json"
CROPPED_GLYPHS_DIR = OUTPUT_DIR / "2_cropped_glyphs"
ORGANIZED_GLYPHS_DIR = OUTPUT_DIR / "3_organized_by_tm"

FINAL_MANIFEST_CSV = OUTPUT_DIR / "3_manifest/final_manifest_sanitized.csv"
SPLITS_DIR = OUTPUT_DIR / "4_splits_sanitized"
TEST_SPLIT_CSV = SPLITS_DIR / "test.csv"
TEST_EMBEDDINGS_NPZ = OUTPUT_DIR / "5_embeddings/test_embeddings_sanitized.npz"


# --

def run_command(command, stage_name, output_artifacts, resume=False):
    """
    Helper function to run a command, stream its output, and handle resuming.

    Args:
        command (list): The command to execute.
        stage_name (str): The name of the pipeline stage.
        output_artifacts (list): A list of Path objects that this stage is expected to create.
        resume (bool): If True, skip this stage if all artifacts already exist.
    """
    print(f"\n{'=' * 25}\n▶️  CHECKING STAGE: {stage_name}\n{'=' * 25}")

    # RESUME LOGIC 
    if resume:
        all_artifacts_exist = all(artifact.exists() for artifact in output_artifacts)
        if all_artifacts_exist:
            print(f"   ⏩ SKIPPING Stage '{stage_name}': Output artifacts already exist.")
            for artifact in output_artifacts:
                print(f"      - Found: {artifact}")
            return True  # Pretend it succeeded so the pipeline can continue
    # END RESUME LOGIC 

    print(f"   RUNNING: {' '.join(command)}\n")
    try:
        process = subprocess.run(
            command,
            check=True,
            text=True,
            encoding='utf-8'
        )
        print(f"\nSUCCESS: Stage '{stage_name}' complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Stage '{stage_name}' failed with exit code {e.returncode}.")
        return False
    except FileNotFoundError:
        print(f"\nERROR: Command not found. Is '{command[0]}' executable and in your PATH?")
        return False


def main(args):
    """Orchestrates the entire pipeline from prediction to evaluation."""
    print(f"🚀 Starting pipeline run for experiment: {EXPERIMENT_NAME}")
    if args.resume:
        print("   Resuming mode is ON. Will skip completed stages.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    should_resume = args.resume

    # STAGE 1: Sliced Prediction
    if not run_command(
            ["python", "predict_patched_yolo.py", "--weights", YOLO_WEIGHTS_PATH, "--source_dir",
             CLEAN_IMAGES_DIR, "--project_name", str(OUTPUT_DIR / "1_predictions"), "--conf", "0.3"],
            "Sliced Prediction", [PREDICTIONS_JSON], resume=should_resume
    ): sys.exit("Pipeline aborted.")

    # STAGE 2: Crop Glyphs
    # Check for the output directory itself as the artifact
    if not run_command(
            ["python", "batch_cropper.py", "--images_dir", CLEAN_IMAGES_DIR, "--predictions_path",
             str(PREDICTIONS_JSON), "--output_dir", str(CROPPED_GLYPHS_DIR), "--train_json", METADATA_JSON_PATH],
            "Crop Glyphs", [CROPPED_GLYPHS_DIR], resume=should_resume
    ): sys.exit("Pipeline aborted.")

    # STAGE 3: Organize & Manifest
    if not run_command(
            ["python", "glyph_organizer.py",
             "--input_dir", str(CROPPED_GLYPHS_DIR),
             "--output_dir", str(ORGANIZED_GLYPHS_DIR),  # USE THE CORRECT DYNAMIC PATH
             "--metadata_json", METADATA_JSON_PATH,
             "--metadata_excel", METADATA_EXCEL_PATH],
            "Organize Glyphs", [ORGANIZED_GLYPHS_DIR], resume=should_resume
    ): sys.exit("Pipeline aborted.")

    if not run_command(
            ["python", "create_full_manifest.py",
             "--merged_dir", str(ORGANIZED_GLYPHS_DIR),  # USE THE CORRECT DYNAMIC PATH
             "--output_csv", str(FINAL_MANIFEST_CSV),
             "--metadata_json", METADATA_JSON_PATH],
            "Create Manifest", [FINAL_MANIFEST_CSV], resume=should_resume
    ): sys.exit("Pipeline aborted.")

    # STAGE 4: Apply Splits
    if not run_command(
            ["python", "create_page_indep_data.py", "--new_manifest_path", str(FINAL_MANIFEST_CSV),
             "--original_train_split_path", ORIGINAL_TRAIN_SPLIT, "--original_test_split_path", ORIGINAL_TEST_SPLIT,
             "--output_dir", str(SPLITS_DIR)],
            "Apply Splits", [TEST_SPLIT_CSV], resume=should_resume
    ): sys.exit("Pipeline aborted.")

    # STAGE 5: Extract Embeddings
    if not run_command(
            ["python", "extract_embeddings.py", "--checkpoint_path", STYLE_ENCODER_PATH, "--manifest_csv",
             str(TEST_SPLIT_CSV), "--output_path", str(TEST_EMBEDDINGS_NPZ)],
            "Extract Embeddings", [TEST_EMBEDDINGS_NPZ], resume=should_resume
    ): sys.exit("Pipeline aborted.")

    # STAGE 6: Evaluate Final Accuracy (We always run this last step, no resume needed)
    if not run_command(
            ["python", "eval_mAP.py", "--test_embedding_path", str(TEST_EMBEDDINGS_NPZ)],
            "Evaluate mAP", [], resume=False  # Always evaluate
    ): sys.exit("Pipeline aborted.")

    print(f"\n🎉 Pipeline for experiment '{EXPERIMENT_NAME}' completed successfully!")
    print(f"   Final results are in the last step's output.")
    print(f"   All artifacts are stored in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    # COMMAND-LINE ARGUMENT PARSING 
    parser = argparse.ArgumentParser(description="Run the full writer identification pipeline.")
    parser.add_argument(
        "--resume",
        action="store_true",  # This makes it a flag, e.g., `python run_pipeline.py --resume`
        help="Resume the pipeline, skipping stages with existing output artifacts."
    )
    args = parser.parse_args()
    main(args)