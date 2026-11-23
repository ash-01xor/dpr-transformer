import csv
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from src.data_utils.corpus import Corpus
from src.evaluation.metrics import Evaluate
from src.retrievers.bm25 import BM25
from src.retrievers.tf_idf import TFIDF
from src.retrievers.utils import retrieveQueryAndGetRelevancies


def evaluate(cfg, corpus, retriever, training_run=None):
    queries = corpus.queries(cfg.split)
    cutoffs = list(cfg.cutoffs)
    hits = retrieveQueryAndGetRelevancies(retriever, queries, max(cutoffs), cfg.batch_size)
    scores = Evaluate(hits, queries, cutoffs)
    settings = {
        "retriever": OmegaConf.to_container(cfg.retriever, resolve=True),
        "batch_size": cfg.batch_size,
        "device": cfg.device,
    }
    if training_run is not None:
        settings["training"] = {
            key: training_run[key] for key in ("training", "model", "selection", "training_dtype")
        }
    common = {
        "dataset": cfg.dataset.name,
        "method": cfg.retriever.method,
        "split": cfg.split,
        "queries": len(queries),
        "passages": len(corpus.all_documents),
        "seed": cfg.seed,
        "configuration": json.dumps(settings, sort_keys=True),
    }
    rows = [dict(common, k=k, **metrics) for k, metrics in scores.items()]
    path = Path(cfg.output_dir) / cfg.dataset.name / cfg.retriever.method / f"{cfg.split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{cfg.dataset.name} {cfg.retriever.method} {cfg.split} ({len(queries)} queries): {scores}")
    return rows


@hydra.main(config_path="config", config_name="fiqa_experiment", version_base=None)
def main(cfg):
    corpus = Corpus(cfg.dataset.name, str(Path(cfg.base_path) / "retrieval_data"), cfg.seed)
    if cfg.action == "train" and cfg.retriever.method == "dpr":
        from src.training.trainer import train

        train(cfg, corpus)
        return
    if cfg.action == "attention" and cfg.retriever.method == "dpr":
        from src.evaluation.attention import attention_report
        from src.retrievers.dense_index import load_dense_retriever

        retriever, _ = load_dense_retriever(cfg, corpus)
        attention_report(cfg, corpus, retriever)
        return
    if cfg.action != "evaluate":
        raise ValueError(f"Unsupported action for {cfg.retriever.method}: {cfg.action}")
    training_run = None
    if cfg.retriever.method == "tf_idf":
        retriever = TFIDF(corpus.all_documents)
    elif cfg.retriever.method == "bm25":
        retriever = BM25(corpus.all_documents, k1=cfg.retriever.k1, b=cfg.retriever.b)
    elif cfg.retriever.method == "dpr":
        from src.retrievers.dense_index import load_dense_retriever

        retriever, training_run = load_dense_retriever(cfg, corpus)
    else:
        raise ValueError(f"Unsupported retriever: {cfg.retriever.method}")
    evaluate(cfg, corpus, retriever, training_run)


if __name__ == "__main__":
    main()
