import pandas as pd

CSV_PATH = "/home/durrieu/Downloads/BallPushing_TNTScreen - TNT_Screen_Lines.csv"
OUTPUT_STEM = "genotypes_table"


def export_vector_table(df: pd.DataFrame, output_stem: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Matplotlib not available ({exc}). Skipping SVG/PDF table export.")
        return

    # Styling inspired by Cell/Nature neuroscience tables
    header_color = "#F2F2F2"
    odd_row_color = "#FFFFFF"
    even_row_color = "#FAFAFA"
    edge_color = "#BDBDBD"

    nrows, ncols = df.shape
    fig_width = max(6, ncols * 2.2)
    fig_height = max(2, nrows * 0.35)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(even_row_color if row % 2 == 0 else odd_row_color)

    fig.tight_layout()
    fig.savefig(f"{output_stem}.svg", format="svg", dpi=300)
    fig.savefig(f"{output_stem}.pdf", format="pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    export_vector_table(df, OUTPUT_STEM)
    print("Outputs: genotypes_table.svg (Illustrator-editable) and genotypes_table.pdf")


if __name__ == "__main__":
    main()
