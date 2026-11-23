import csv
from collections import defaultdict

from datasets import load_dataset
from huggingface_hub import hf_hub_download

def get_fiqa_dataset(base_dir: str):
    corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus", cache_dir=base_dir)
    queries = load_dataset("BeIR/fiqa", "queries", split="queries", cache_dir=base_dir)
    judgments = {}
    for split in ("train", "dev", "test"):
        path = hf_hub_download(
            "BeIR/fiqa-qrels", f"{split}.tsv", repo_type="dataset",
            cache_dir=base_dir,
        )
        with open(path, encoding="utf-8", newline="") as source:
            positives = defaultdict(list)
            for row in csv.DictReader(source, delimiter="\t"):
                if int(row["score"]) > 0:
                    positives[row["query-id"]].append(row["corpus-id"])
        judgments[split] = positives
    return corpus, queries, judgments

def get_squad_dataset(base_dir: str):
    return load_dataset("rajpurkar/squad", cache_dir=base_dir)
