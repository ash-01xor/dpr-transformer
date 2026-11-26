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

The top-level `seed` controls data splitting. `training.seed` controls PyTorch initialization, dropout, batch shuffling, and positive-passage sampling. It defaults to the top-level seed when omitted or set to `null`. Checkpoint metadata records the effective training seed. To vary training randomness on a fixed split, use `seed=42 training.seed=43` and a separate `output_dir` for each run. Evaluation uses the original split seed, for example `seed=42`, when loading that run.

Runs configured for more than 20 epochs also save `checkpoint_latest.pt` at epoch 20 and replace it every 20 epochs. Evaluation loads the best checkpoint, `checkpoint.pt`, by default. Use a distinct `output_dir` for each experiment to keep its checkpoints and metrics together.

## Held-out evaluation and attention

```bash
python main.py -m dataset=fiqa,squad retriever=tf_idf,bm25 split=test

for dataset in fiqa squad; do
  python main.py dataset="$dataset" retriever=dpr split=test device=cuda
  python main.py dataset="$dataset" retriever=dpr action=attention split=test device=cuda
done

python -m src.evaluation.results
```

The first DPR evaluation builds the FAISS inner-product index and reloads it for search. Subsequent runs reuse the saved index for that checkpoint. The results command writes `outputs/retrieval/test_results.csv` with Precision@k, Recall@k, and MRR@k for all six comparisons at cutoffs 1, 5, 10, and 20.

Attention analysis saves examples, attention weights, and heatmaps from the requested split. By default it selects two queries with a judged relevant passage in the top ten and two without one. Use `attention.k` and `attention.examples_per_outcome` to change the selection.

The saved arrays retain every layer and head. To plot individual heads, pass the array file and one-based layer/head numbers. Repeat the panel option to compare heads in one figure.

```bash
python -m src.evaluation.plot_attention \
  --panel ARRAY_FILE 4 2 --panel ARRAY_FILE 4 4 \
  --output FIGURE_PREFIX
```

This writes a PDF and PNG using the same colour scale for each panel.

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
