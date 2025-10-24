## End-to-End Glyph Detection and Writer Identification

### 1. Project Overview

This project implements a complete pipeline for writer identification in ancient Greek manuscripts. It proceeds from raw, high-resolution page images to a final, quantitative analysis of scribal style.

The core methodology involves two main stages:
1.  **Glyph Detection:** A YOLOv8-Medium model, enhanced with Sliced Aided Hyper Inference (SAHI), is used to detect and crop individual characters (glyphs) from manuscript pages.
2.  **Style Feature Extraction:** A ResNet-18 encoder, trained using a **Supervised Contrastive Learning** framework, converts each glyph into a 128-dimensional embedding that captures its unique stylistic properties.

This approach has proven to be highly effective, achieving **85.89% mAP** in a standard writer retrieval task and a state-of-the-art **97.91% accuracy** in a zero-shot (writer-independent) identification task.

### 2. Prerequisites and Setup

#### A. Environment Setup
It is recommended to use a Python virtual environment. This project requires Python 3.8+.

```bash
# Clone the repository (if applicable)
# git clone ...
# cd project_directory

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate    # On Windows

# Install the required packages
pip install torch torchvision pandas numpy ultralytics sahi scikit-learn matplotlib seaborn tqdm pillow openpyxl
```


### 3. Running the Full Pipeline: From Raw Images to Final Results

This is the step-by-step guide to replicate the main results of the project. Each step is essential and must be run in order.



#### **Step 1: Data Preparation & Standardization**

**Purpose:** To clean the raw dataset by converting all images to a standard JPG format, fixing corrupted files, and preserving the directory structure. This ensures the dataset is uniform and error-free for all subsequent steps.

**Command:**
```bash
python scripts/standardize_images.py
```

**Expected Output:**
*   A new directory named `datasets/HomerComp_Cleaned/` will be created, containing the sanitized and standardized full-page manuscript images.



#### **Step 2: Train the Glyph Detector (YOLOv8)**

**Purpose:** To train the YOLOv8-Medium model that will be used to find glyphs. This involves converting the COCO annotations to YOLO format first.

**Commands:**
```bash
# 1. Convert COCO annotations to YOLO format
python scripts/convert_coco_to_yolo.py

# 2. Create the YAML configuration file for YOLO training
python scripts/create_yolo_yaml.py

# 3. Run the training process (this will take a significant amount of time)
python scripts/run_yolo.py train --model_start yolov8m.pt --epochs 100 --batch 8 --project_name yolo_training_run --imgsz 512
```

**Expected Output:**
*   A directory `datasets/yolo_glyphs/` with the training data in YOLO format.
*   A configuration file `datasets/yolo_glyphs.yaml`.
*   A directory `yolo_training_run/` containing the training results. The most important file is the trained model weights at `yolo_training_run/train/weights/best.pt`.



#### **Step 3: Detect and Crop All Glyphs**

**Purpose:** Use the trained YOLOv8 model with SAHI to perform inference on all standardized manuscript pages and then crop the detected glyphs into individual image files.

**Commands:**
```bash
# 1. Run sliced prediction on all clean images
python scripts/predict_patched_yolo.py --weights yolo_training_run/train/weights/best.pt --source_dir datasets/HomerComp_Cleaned/ --project_name pipeline_output

# 2. Crop the glyphs based on the prediction JSON
python scripts/batch_cropper.py --images_dir datasets/HomerComp_Cleaned/ --predictions_path pipeline_output/predict/predictions_by_filename.json --output_dir pipeline_output/2_cropped_glyphs --train_json papytwin/HomerCompTraining/HomerCompTraining.json
```

**Expected Output:**
*   A JSON file at `pipeline_output/predict/predictions_by_filename.json` with all detected bounding boxes.
*   A directory `pipeline_output/2_cropped_glyphs/` containing tens of thousands of individual cropped glyph images.



#### **Step 4: Build the Final, Sanitized Glyph Dataset**

**Purpose:** Organize the raw crops by writer, remove contamination artifacts, and create the final, analysis-ready manifest files and train/test splits. This is a critical data integrity stage.

**Commands:**
```bash
# 1. Organize flat crops into folders by writer ID
python scripts/glyph_organizer.py --input_dir pipeline_output/2_cropped_glyphs --output_dir pipeline_output/3_organized_by_tm --metadata_json papytwin/HomerCompTraining/HomerCompTraining.json --metadata_excel papytwin/1.CompetitionOverview.xlsx

# 2. Sanitize the organized dataset to remove modern character artifacts
python scripts/sanitize_dataset.py --source_dir pipeline_output/3_organized_by_tm --dest_dir pipeline_output/4_sanitized_glyphs

# 3. Create the master manifest from the sanitized glyphs
python scripts/create_full_manifest.py --merged_dir pipeline_output/4_sanitized_glyphs --output_csv pipeline_output/5_final_manifest.csv --metadata_json papytwin/HomerCompTraining/HomerCompTraining.json

# 4. Create the final page-independent train/test splits
python scripts/prep_final_dataset.py
```

**Expected Output:**
*   Directories `pipeline_output/3_organized_by_tm/` and `pipeline_output/4_sanitized_glyphs/`.
*   A master CSV `pipeline_output/5_final_manifest.csv`.
*   A directory `pipeline_output/6_final_splits` containing `train_final.csv` and `test_final.csv`, which are the official splits for the main experiment.



#### **Step 5: Train the Style Encoder**

**Purpose:** To train the core model of the project. This script uses the sanitized, page-independent training split to learn the discriminative style embeddings via Supervised Contrastive Learning.

**Command:**
```bash
python scripts/train_supervised.py
```

**Expected Output:**
*   The trained style encoder weights saved at `checkpoints/supcon_encoder.pt`.



#### **Step 6: Extract Embeddings for the Test Set**

**Purpose:** Use the trained style encoder to generate the 128-dimensional feature vectors for every glyph in the held-out test set.

**Command:**
```bash
python scripts/extract_embeddings.py --checkpoint_path checkpoints/supcon_encoder.pt --manifest_csv pipeline_output/6_final_splits/test_final.csv --output_path pipeline_output/7_test_embeddings.npz
```

**Expected Output:**
*   An `.npz` file at `pipeline_output/7_test_embeddings.npz` containing the embeddings and paths for the test set.



#### **Step 7: Run Final Evaluation (mAP)**

**Purpose:** To calculate the primary writer identification result on the test set embeddings.

**Command:**
```bash
python scripts/eval_mAP.py --test_embedding_path pipeline_output/7_test_embeddings.npz
```

**Expected Final Result:**
*   The script will print the final score to the console.



### 4. Running Secondary & Analytical Scripts

After completing the main pipeline, you can run the following experiments.

#### A. Writer-Independent (Zero-Shot) Evaluation

**Purpose:** To evaluate the system's ability to identify unseen writers, demonstrating the generalization power of the learned features.

**Commands:**
```bash
# (Assuming the full embeddings for ALL sanitized glyphs are extracted first)
# You may need a script to extract embeddings for the full manifest, not just the test split.

# 1. Create the writer-disjoint splits
python scripts/create_writer_disjoint_split.py --full_manifest pipeline_output/5_final_manifest.csv --output_dir pipeline_output/8_writer_disjoint_splits

# 2. Run the leave-one-image-out evaluation on the test set
python scripts/extract_embeddings.py --checkpoint_path checkpoints/supcon_encoder.pt --manifest_csv pipeline_output/8_writer_disjoint_splits/test_writer_disjoint.csv --output_path pipeline_output/8_writer_disjoint_splits/test_embeddings_disjoint.npz
    
# (This script also requires embeddings for the writer-disjoint test set)
python scripts/eval_writer_disjoint_loio.py --embeddings_path pipeline_output/8_writer_disjoint_splits/test_embeddings_disjoint.npz --manifest_path pipeline_output/8_writer_disjoint_splits/test_writer_disjoint.csv
```
**Expected Final Result:**
*   The script will print a top-1 page-level accuracy score.

#### B. VLAC Comparative Experiment

**Purpose:** To compare the primary glyph-level method against a standard page-level aggregation technique (VLAC).

**Commands:**
```bash
# 1. Generate VLAC page descriptors

python scripts/extract_embeddings.py --checkpoint_path checkpoints/supcon_encoder.pt --manifest_csv pipeline_output/5_final_manifest.csv --output_path pipeline_output/full_embeddings.npz
    
python scripts/run_simple_vlac.py --embeddings pipeline_output/full_embeddings.npz --manifest pipeline_output/5_final_manifest.csv --output_path pipeline_output/9_vlac_descriptors.npz

# 2. Evaluate the mAP on the VLAC descriptors
python scripts/eval_page_level_map.py --descriptors_path pipeline_output/9_vlac_descriptors.npz
```
**Expected Final Result:**
*   The script will print a page-level mAP score.

#### C. Paleographic Analysis & Visualization

**Purpose:** To use the learned embeddings to generate novel paleographic insights. These scripts are the "payoff" of the project. (Run after extracting embeddings for the full dataset).

**Example Commands:**
```bash
# Analyze writer distinctiveness for the character 'epsilon'
python scripts/analyze_char_distinctiveness.py --embeddings path/to/full_embeddings.npz --manifest pipeline_output/5_final_manifest.csv --character epsilon --output epsilon_distinctiveness.csv

# Visualize the discovered allographs for 'epsilon'
python scripts/visualize_allographs.py --embeddings path/to/full_embeddings.npz --manifest pipeline_output/5_final_manifest.csv --character epsilon --output epsilon_allographs.png
```




### **GRK Leave one out**

#### **Step 1: Glyph Detection on GRK Test Images**

**Purpose:** Run your YOLOv8 model with SAHI on the images located **only in the `grk_dataset/test/` directory.**

**Command:**
```bash
python scripts/predict_patched_yolo_final.py --weights yolo_training_run/train/weights/best.pt --source_dir grk_dataset/test/ --project_name grk_pipeline
```

**Expected Output:** A `grk_pipeline/predict/predictions_by_filename.json` file containing detections for the test images only.



#### **Step 2: Crop Glyphs from GRK Test Images**

**Purpose:** Crop the individual glyphs from the original test set images.

**Command:**
```bash
python scripts/batch_cropper_generic.py --images_dir grk_dataset/test/ --predictions_path grk_pipeline/predict/predictions_by_filename.json --output_dir grk_pipeline/2_cropped_glyphs
```
**Expected Output:** A `grk_pipeline/2_cropped_glyphs/` directory containing only glyphs cropped from the test images.



#### **Step 3: Create the Manifest for GRK *Test* Glyphs**

**Purpose:** To create a master CSV that links every cropped test set glyph to its ground-truth writer ID. We will use an additional argument to ensure we only pull writer information for the images that are actually in the `test` folder, preventing any data leakage from the `train` set.

**Command:**
```bash
python scripts/create_manifest_generic.py --glyph_dir grk_pipeline/2_cropped_glyphs/ --gt_csv grk_dataset/grk_truth.csv --output_csv grk_pipeline/3_grk_manifest.csv --image_list_dir grk_dataset/test/
```
**Expected Output:** A manifest file `grk_pipeline/3_grk_manifest.csv` containing paths and writer IDs for the test set glyphs only.



#### **Step 4: Extract Style Embeddings for GRK Test Glyphs**

**Purpose:** Generate the 128-dimensional feature vectors for every cropped test glyph using your HomerComp-trained model.

**Command:**
```bash
python scripts/extract_embeddings.py --checkpoint_path checkpoints/supcon_encoder.pt --manifest_csv grk_pipeline/3_grk_manifest.csv --output_path grk_pipeline/4_grk_embeddings.npz
```
**Expected Output:** An `.npz` file at `grk_pipeline/4_grk_embeddings.npz`.



#### **Step 5: Run the Final Evaluation (Leave-One-Image-Out CV)**

**Purpose:** Perform the final leave-one-image-out cross-validation on the page prototypes derived from the GRK test set glyphs.

**Command:**
```bash
python scripts/eval_leave_one_out.py --embeddings_path grk_pipeline/4_grk_embeddings.npz --manifest_path grk_pipeline/3_grk_manifest.csv
```

**Expected Final Result:**
The script will print the final zero-shot Top-1 accuracy to the console.