import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def plot_distinctiveness(args):
    """
    Creates a horizontal bar chart to visualize the most stylistically
    distinct writers for a given character.
    """
    csv_path = args.csv_path
    top_n = args.top_n
    character_name = args.character_name
    output_path = args.output_path

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    # Sort by score and take the top N
    df_top = df.sort_values(by="Distinctiveness_Score", ascending=False).head(top_n)

    # Create the Plot 
    plt.style.use('seaborn-talk')  # Use a nice style for presentations
    plt.figure(figsize=(12, 8))

    # Create a color palette that goes from dark to light (e.g., Viridis, Plasma)
    palette = sns.color_palette("rocket_r", n_colors=top_n)

    ax = sns.barplot(
        x="Distinctiveness_Score",
        y="Writer_ID",
        data=df_top,
        palette=palette,
        orient='h'
    )

    plt.title(f"Top {top_n} Most Stylistically Distinct Writers for '{character_name.capitalize()}'", fontsize=18,
              pad=20)
    plt.xlabel("Distinctiveness Score (Higher is More Unique vs. Global Average)", fontsize=14, labelpad=15)
    plt.ylabel("Writer ID", fontsize=14, labelpad=15)

    # Invert y-axis to have the top writer at the top
    ax.invert_yaxis()

    # Add data labels to the bars for clarity
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.005, p.get_y() + p.get_height() / 2.,
                f'{width:.3f}',
                ha='left', va='center', fontsize=11)

    plt.xlim(0, df_top['Distinctiveness_Score'].max() * 1.1)  # Add some padding to the x-axis

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Distinctiveness plot saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize writer distinctiveness from a CSV file.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the distinctiveness CSV file.")
    parser.add_argument("--top_n", type=int, default=15, help="Number of top writers to display.")
    parser.add_argument("--character_name", type=str, default="Epsilon",
                        help="Name of the character for the plot title.")
    parser.add_argument("--output_path", type=str, default="distinctiveness_plot.png",
                        help="Path to save the output plot.")
    args = parser.parse_args()
    plot_distinctiveness(args)