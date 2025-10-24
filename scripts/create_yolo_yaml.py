import yaml
import json
from pathlib import Path

json_path = Path("papytwin/HomerCompTraining/HomerCompTraining.json")
output_yaml_path = Path("datasets/yolo_glyphs.yaml")

print(f"Loading category data from {json_path}...")
with open(json_path) as f:
    coco_data = json.load(f)

# Sort categories by ID to ensure correct order
categories = sorted(coco_data['categories'], key=lambda x: x['id'])
class_names = [cat['name'] for cat in categories]

# Define the dataset structure for the YAML file
yolo_yaml_data = {
    'path': str(Path('datasets/yolo_glyphs').resolve()), # Absolute path to data root
    'train': 'images/train',
    'val': 'images/train',
    'nc': len(class_names),
    'names': class_names
}

print(f"Saving YAML configuration to {output_yaml_path}...")
with open(output_yaml_path, 'w') as f_yaml:
    yaml.dump(yolo_yaml_data, f_yaml, sort_keys=False)

print("\nYAML file created successfully.")
print("YAML Content ")
print(yaml.dump(yolo_yaml_data, sort_keys=False))