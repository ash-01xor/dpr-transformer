# 02456 Retrieval Project

Compare TF-IDF, BM25, and Dense Passage Retrieval on the same corpus and held-out queries for FiQA and SQuAD. DPR uses two randomly initialized Transformer encoders and learns from query–passage relevance.

## Setup

Use Python 3.10 or newer. Install the CUDA build of PyTorch recommended for your GPU before installing the remaining dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first run downloads the complete BEIR FiQA corpus and its original relevance judgments, or SQuAD 1.1. Prepared data are reused from `retrieval_data/prepared/`. For SQuAD, seed 42 holds out 10% of the original training articles for development and the published validation split serves as the test set.

## Development and checkpoint selection

```bash
python main.py -m dataset=fiqa,squad retriever=tf_idf,bm25 split=dev

for dataset in fiqa squad; do
  python main.py dataset="$dataset" retriever=dpr action=train device=cuda
done
```

DPR learns a WordPiece vocabulary from training questions and their relevant passages. The trainer saves the epoch with the highest development MRR@10, both encoders, the tokenizer, and the settings under `outputs/retrieval/<dataset>/dpr/`.

## Held-out evaluation and attention

```bash
python main.py -m dataset=fiqa,squad retriever=tf_idf,bm25 split=test

for dataset in fiqa squad; do
  python main.py dataset="$dataset" retriever=dpr split=test device=cuda
  python main.py dataset="$dataset" retriever=dpr action=attention split=test device=cuda
done

python -m src.evaluation.results
```

The first DPR evaluation builds the FAISS inner-product index and reloads it for search. Subsequent runs reuse the saved index for that checkpoint. The results command writes `outputs/retrieval/test_results.csv` with Precision@k, Recall@k, and MRR@k for all six comparisons at cutoffs 1, 5, 10, and 20. Each dataset's DPR `attention/` directory contains token-labeled PDFs for a hit and a miss at rank 10. The report labels development examples when the test split lacks either outcome.

Run the encoder, training, and persistence checks with `python -m pytest -q` after installing pytest.

## DTU HPC batch jobs

From the repository root on the login node, prepare the job log directory and submit the scripts.

```bash
mkdir -p outputs/retrieval/jobs
bsub < scripts/sparse.lsf
bsub < scripts/dpr-fiqa.lsf
bsub < scripts/dpr-squad.lsf
bjobs
```

The DPR scripts train with the project `.venv` and the Hydra defaults in `config/`. To adjust a setting for one job, append its Hydra override to the Python command in that script, for example `training.epochs=6`.

After training, run the held-out evaluation and attention commands above. Once all comparisons finish, `bsub < scripts/results.lsf` assembles the results table.

[DTU batch job guide](https://www.hpc.dtu.dk/?page_id=1416) · [DTU GPU job guide](https://www.hpc.dtu.dk/?page_id=4976)
