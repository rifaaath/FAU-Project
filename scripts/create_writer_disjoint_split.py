import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse


def create_split(args):
    manifest_path = Path(args.full_manifest)
    output_dir = Path(args.output_dir)
    test_size = args.test_size

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading full manifest from: {manifest_path}")
    df = pd.read_csv(manifest_path)

    # Get the list of all unique writers
    all_writers = df['tm_id'].unique()
    print(f"Found {len(all_writers)} unique writers in the dataset.")

    # Split the list of writers into a training set and a test set
    train_writers, test_writers = train_test_split(
        all_writers,
        test_size=test_size,
        random_state=42
    )

    print(f"Splitting writers: {len(train_writers)} for training, {len(test_writers)} for testing.")

    # Create the final train/test DataFrames by filtering the main manifest
    df_train = df[df['tm_id'].isin(train_writers)]
    df_test = df[df['tm_id'].isin(test_writers)]

    # Save the new split manifests
    train_output_path = output_dir / "train_writer_disjoint.csv"
    test_output_path = output_dir / "test_writer_disjoint.csv"

    df_train.to_csv(train_output_path, index=False)
    df_test.to_csv(test_output_path, index=False)

    print("\nSummary ")
    print(f"Total glyphs: {len(df)}")
    print(f"Training glyphs (from {len(train_writers)} writers): {len(df_train)}")
    print(f"Test glyphs (from {len(test_writers)} writers): {len(df_test)}")
    print(f"\nWriter-disjoint splits saved to: '{output_dir.resolve()}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a writer-disjoint (writer-independent) train/test split.")
    parser.add_argument("--full_manifest", type=str, required=True,
                        help="Path to the complete, sanitized manifest CSV.")
    parser.add_argument("--output_dir", type=str, default="writer_disjoint_splits",
                        help="Directory to save the split files.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of writers to allocate to the test set.")
    args = parser.parse_args()
    create_split(args)