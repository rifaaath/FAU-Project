# Set the name of your final, successful experiment run
export EXPERIMENT_NAME="yolo_medium_final_run_v4"

# Set the name for the output directory for this analysis run
export ANALYSIS_OUTPUT_DIR="analysis_results_v4"

# Define the paths to your definitive dataset artifacts
export FULL_EMBEDDINGS_NPZ="pipeline_runs/${EXPERIMENT_NAME}/full_dataset_embeddings.npz"
export FULL_MANIFEST_CSV="pipeline_runs/${EXPERIMENT_NAME}/3_manifest/final_manifest.csv"
export TEST_EMBEDDINGS_NPZ="pipeline_runs/${EXPERIMENT_NAME}/5_embeddings/test_embeddings.npz"
export TEST_MANIFEST_CSV="pipeline_runs/${EXPERIMENT_NAME}/4_splits/test.csv"
export OVERALL_MAP_SCORE="85.89"

# SETUP
# Create the output directory for the results
mkdir -p $ANALYSIS_OUTPUT_DIR
echo "Analysis results will be saved in '${ANALYSIS_OUTPUT_DIR}'"

# Generate embeddings for the full dataset if they don't exist yet
if [ ! -f "$FULL_EMBEDDINGS_NPZ" ]; then
    echo "Generating embeddings for the full dataset (this may take a minute)... "
    python extract_embeddings.py \
        --checkpoint_path "checkpoints/supcon_encoder.pt" \
        --manifest_csv "$FULL_MANIFEST_CSV" \
        --output_path "$FULL_EMBEDDINGS_NPZ"
    echo "Full dataset embeddings created. "
else
    echo "Full dataset embeddings already exist. Skipping generation. "
fi


# ANALYSIS 1: WRITER DISTINCTIVENESS
echo -e "\nRunning Analysis 1: Writer Distinctiveness (Epsilon) "
python analyze_char_distinctiveness.py \
    --embeddings "$FULL_EMBEDDINGS_NPZ" \
    --manifest "$FULL_MANIFEST_CSV" \
    --character "epsilon" \
    --output "${ANALYSIS_OUTPUT_DIR}/distinctiveness_epsilon.csv"

echo -e "\nRunning Analysis 1: Writer Distinctiveness (Alpha) "
python analyze_char_distinctiveness.py \
    --embeddings "$FULL_EMBEDDINGS_NPZ" \
    --manifest "$FULL_MANIFEST_CSV" \
    --character "alpha" \
    --output "${ANALYSIS_OUTPUT_DIR}/distinctiveness_alpha.csv"


# ANALYSIS 2: WRITER CONSISTENCY VS. DISTANCE
echo -e "\nRunning Analysis 2: Writer Consistency vs. Distance "
# First, find the best character pair to use.
echo "Finding the best character pair for consistency analysis..."
BEST_PAIR_INFO=$(python find_char_pairs.py --manifest "$FULL_MANIFEST_CSV" --min_samples 5 | grep "'")
CHAR_A=$(echo $BEST_PAIR_INFO | cut -d"'" -f2)
CHAR_B=$(echo $BEST_PAIR_INFO | cut -d"'" -f4)
echo "Found best pair: '$CHAR_A' and '$CHAR_B'. Running analysis..."

python analyze_writer_consistency.py \
    --embeddings "$FULL_EMBEDDINGS_NPZ" \
    --manifest "$FULL_MANIFEST_CSV" \
    --char_a "$CHAR_A" \
    --char_b "$CHAR_B" \
    --output "${ANALYSIS_OUTPUT_DIR}/consistency_vs_distance.csv"


# ANALYSIS 3: ALLOGRAPH VISUALIZATION
echo -e "\nRunning Analysis 3: Allograph Visualization (Epsilon) "
python visualize_allographs.py \
    --embeddings "$FULL_EMBEDDINGS_NPZ" \
    --manifest "$FULL_MANIFEST_CSV" \
    --character "epsilon" \
    --n_clusters 16 \
    --output "${ANALYSIS_OUTPUT_DIR}/allographs_epsilon.png"


# ANALYSIS 4: WRITER SIMILARITY NETWORK
echo -e "\nRunning Analysis 4: Writer Similarity Heatmap "
python analyze_writer_similarity.py \
    --embeddings "$FULL_EMBEDDINGS_NPZ" \
    --manifest "$FULL_MANIFEST_CSV" \
    --min_samples 20 \
    --output "${ANALYSIS_OUTPUT_DIR}/writer_similarity_heatmap.png"


# ANALYSIS 5: CHARACTER DIFFICULTY / STYLISTIC RICHNESS
echo -e "\nRunning Analysis 5: Character Difficulty "
python analyze_char_difficulty.py \
    --embeddings "$TEST_EMBEDDINGS_NPZ" \
    --manifest "$TEST_MANIFEST_CSV" \
    --output_csv "${ANALYSIS_OUTPUT_DIR}/character_difficulty.csv" \
    --output_plot "${ANALYSIS_OUTPUT_DIR}/character_difficulty_plot.png" \
    --overall_map "$OVERALL_MAP_SCORE"

echo -e "\n\n🎉 All final analyses completed successfully!"
echo "   Results are saved in the '${ANALYSIS_OUTPUT_DIR}' directory."