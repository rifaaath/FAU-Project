import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np


def plot_consistency_vs_distance(args):
    """
    Creates a scatter plot to visualize writer consistency vs. inter-letter distance.
    """
    csv_path = args.csv_path
    char_a = args.char_a
    char_b = args.char_b
    output_path = args.output_path

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    # Prepare Data for Plotting 
    consistency_col = f"{char_a.capitalize()}_Cohesion"
    distance_col = f"{char_a.capitalize()}_{char_b.capitalize()}_Distance"
    size_col = f"Num_{char_a.capitalize()}"  # Use number of samples for point size

    # Create the Plot 
    plt.style.use('seaborn-talk')
    plt.figure(figsize=(14, 10))

    ax = sns.scatterplot(
        x=distance_col,
        y=consistency_col,
        size=size_col,
        sizes=(50, 500),  # Min and max pt sizes
        hue=distance_col,  # Color points by distance
        palette="viridis",
        data=df,
        legend="brief"  # compact legend
    )

    # Add Labels and Titles 
    plt.title(f"Writer Analysis: Intra-Letter Consistency vs. Inter-Letter Distance", fontsize=20, pad=20)
    plt.xlabel(
        f"Inter-Letter Distance (Higher = '{char_a.capitalize()}' and '{char_b.capitalize()}' are more different)",
        fontsize=14, labelpad=15)
    plt.ylabel(f"Intra-Letter Cohesion (Higher = '{char_a.capitalize()}'s are more consistent)", fontsize=14,
               labelpad=15)

    # Add Annotations for Quadrants 
    x_mean = df[distance_col].mean()
    y_mean = df[consistency_col].mean()
    plt.axvline(x=x_mean, color='grey', linestyle='--', linewidth=0.8)
    plt.axhline(y=y_mean, color='grey', linestyle='--', linewidth=0.8)

    plt.text(df[distance_col].max(), df[consistency_col].max(), 'High Consistency\nHigh Distance\n(Clear & Neat)',
             horizontalalignment='right', verticalalignment='top', fontsize=12, color='darkgreen', alpha=0.7)
    plt.text(df[distance_col].min(), df[consistency_col].max(), 'High Consistency\nLow Distance\n(Neat but Ambiguous)',
             horizontalalignment='left', verticalalignment='top', fontsize=12, color='darkorange', alpha=0.7)
    plt.text(df[distance_col].min(), df[consistency_col].min(), 'Low Consistency\nLow Distance\n(Sloppy & Ambiguous)',
             horizontalalignment='left', verticalalignment='bottom', fontsize=12, color='darkred', alpha=0.7)
    plt.text(df[distance_col].max(), df[consistency_col].min(), 'Low Consistency\nHigh Distance\n(Sloppy but Clear)',
             horizontalalignment='right', verticalalignment='bottom', fontsize=12, color='darkblue', alpha=0.7)

    # Improve legend
    h, l = ax.get_legend_handles_labels()
    # Find the start of the size legend
    size_legend_start = next((i for i, label in enumerate(l) if label == size_col), None)
    if size_legend_start is not None:
        l[size_legend_start] = f"Num. of '{char_a.capitalize()}' Samples"  # Prettier legend title

    ax.legend(h, l, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend

    plt.savefig(output_path, dpi=300)
    print(f"Consistency vs. Distance plot saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize writer consistency from a CSV file.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the consistency vs. distance CSV file.")
    parser.add_argument("--char_a", type=str, default="Upsilon", help="The first character (for y-axis and size).")
    parser.add_argument("--char_b", type=str, default="Alpha", help="The second character (for x-axis).")
    parser.add_argument("--output_path", type=str, default="consistency_plot.png", help="Path to save the output plot.")
    args = parser.parse_args()
    plot_consistency_vs_distance(args)