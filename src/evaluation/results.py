import argparse
import csv
from itertools import product
from pathlib import Path


def collect(root, split):
    rows = []
    for dataset, method in product(("fiqa", "squad"), ("tf_idf", "bm25", "dpr")):
        path = root / dataset / method / f"{split}.csv"
        with path.open(newline="", encoding="utf-8") as source:
            result = list(csv.DictReader(source))
        if {int(row["k"]) for row in result} != {1, 5, 10, 20} or len(result) != 4:
            raise ValueError(f"Expected a single row per cutoff in {path}")
        if any((row["dataset"], row["method"], row["split"]) != (dataset, method, split) for row in result):
            raise ValueError(f"Unexpected dataset, method, or split in {path}")
        rows.extend(result)
    for dataset in ("fiqa", "squad"):
        counts = {
            (row["queries"], row["passages"], row["seed"]) for row in rows if row["dataset"] == dataset
        }
        if len(counts) != 1:
            raise ValueError(f"Results use different data settings for {dataset}")
    output = root / f"{split}_results.csv"
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine the six retrieval comparisons.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retrieval"))
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    options = parser.parse_args()
    print(collect(options.output_dir, options.split))
