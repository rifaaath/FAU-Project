import argparse
import json
from pathlib import Path
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from tqdm import tqdm
import glob


def predict_with_slicing(args):
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Model weights not found at '{weights_path}'")
        return

    print("Starting YOLOv8 w/ SAHI")

    print("Loading model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(weights_path),
        confidence_threshold=args.conf,
        device=args.device
    )

    source_image_dir = Path(args.source_dir)

    # Find all image files recursively
    print(f"Recursively searching for images in '{source_image_dir}'...")
    image_paths = glob.glob(str(source_image_dir / "**/*.[jJ][pP][gG]"), recursive=True)

    if not image_paths:
        print(f"Error: No JPG images found in '{source_image_dir}'. Check the path.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # Manually iterate and store results in a dictionary
    all_results_dict = {}
    total_detections = 0
    for image_path_str in tqdm(image_paths, desc="Predicting on images"):
        image_path = Path(image_path_str)

        result = get_sliced_prediction(
            image=image_path_str,
            detection_model=detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        # Convert predictions to a simple list of dictionaries for this image
        predictions_for_image = []
        if result.object_prediction_list:
            for pred in result.object_prediction_list:
                predictions_for_image.append({
                    "bbox": [int(b) for b in pred.bbox.to_xywh()],  # Use xywh format for consistency
                    "category_id": pred.category.id,
                    "score": pred.score.value
                })

        # Use the image's filename as the key
        all_results_dict[image_path.name] = predictions_for_image
        total_detections += len(predictions_for_image)


    output_dir = Path(args.project_name) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = output_dir / 'predictions_by_filename.json'

    with open(output_json_path, 'w') as f:
        json.dump(all_results_dict, f, indent=4)

    print("\nSliced Prediction Complete ")
    print(f"Found {total_detections} total detections.")
    print(f"Predictions JSON saved to: {output_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run sliced prediction with YOLOv8 and SAHI.")
    parser.add_argument('--weights', type=str, required=True, help="Path to best.pt")
    parser.add_argument('--source_dir', type=str, default='datasets/HomerComp_Cleaned/',
                        help="Root directory of high-res images.")
    parser.add_argument('--project_name', type=str, default='final_yolo_predictions_robust')
    parser.add_argument('--name', type=str, default='predict')
    parser.add_argument('--conf', type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device, e.g., 'cuda:0' or 'cpu'.")

    args = parser.parse_args()
    predict_with_slicing(args)