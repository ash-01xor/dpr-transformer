"""Plot individual attention heads from saved attention arrays."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.evaluation.attention import plot_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", nargs=3, action="append", required=True,
        metavar=("ARRAY", "LAYER", "HEAD"),
        help="Saved .npz file and one-based layer/head numbers. Repeat for multiple panels.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output prefix for the PDF and PNG.")
    parser.add_argument("--title", default=None, help="Title for the complete figure.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--vmax", type=float, default=1.0)
    args = parser.parse_args()

    panels = []
    for filename, layer, head in args.panel:
        try:
            with np.load(filename, allow_pickle=False) as saved:
                data = {"tokens": saved["tokens"], "attention": saved["attention"]}
            weights = data["attention"]
            length = len(data["tokens"])
            if weights.ndim != 4 or weights.shape[-2:] != (length, length):
                raise ValueError("Attention must have axes [layer, head, receiving token, source token].")
            panels.append((data, int(layer), int(head)))
        except (OSError, KeyError, ValueError) as error:
            parser.error(f"{filename}: {error}")

    columns = min(2, len(panels))
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 6 * rows),
                             squeeze=False, constrained_layout=True)
    try:
        for ax, (data, layer, head) in zip(axes.flat, panels):
            plot_attention(data, layer=layer, head=head, max_tokens=args.max_tokens,
                           vmax=args.vmax, ax=ax)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        if args.title:
            fig.suptitle(args.title)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            output = Path(str(args.output) + suffix)
            fig.savefig(output, dpi=200)
            print(output)
    except ValueError as error:
        parser.error(str(error))
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
