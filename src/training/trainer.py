import json
import os
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import torch
from omegaconf import OmegaConf

from src.evaluation.metrics import Evaluate
from src.retrievers.dpr import DPR, fit_tokenizer
from src.training.loss import DPRLoss


def positive_mask(queries, passage_ids):
    return torch.tensor([
        [passage_id in query.relevant_document_ids for passage_id in passage_ids]
        for query in queries
    ], dtype=torch.bool)


def checkpoint_path(cfg):
    return Path(cfg.output_dir) / cfg.dataset.name / "dpr" / "checkpoint.pt"


def train(cfg, corpus):
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    train_queries = corpus.queries("train")
    dev_queries = corpus.queries("dev")
    documents = corpus.all_documents
    document_ids = documents["document_id"]
    row_by_id = {identifier: index for index, identifier in enumerate(document_ids)}
    passages = documents["text"]
    training_ids = {identifier for q in train_queries for identifier in q.relevant_document_ids}
    tokenizer = fit_tokenizer(
        (q.text for q in train_queries),
        (text for identifier, text in zip(document_ids, passages, strict=True) if identifier in training_ids),
        model_config["vocab_size"],
    )
    model = DPR(tokenizer, model_config).to(cfg.device)
    use_bfloat16 = model.device.type == "cuda" and torch.cuda.get_device_capability(model.device)[0] >= 8
    train_tokens = model.tokenize([q.text for q in train_queries])
    dev_tokens = model.tokenize([q.text for q in dev_queries])
    passage_tokens = model.tokenize(passages, passage=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    loss_function = DPRLoss()
    best = float("-inf")
    checkpoint = checkpoint_path(cfg)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []
    progress = list(range(len(train_queries)))
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        rng.shuffle(progress)
        total_loss = 0.0
        batch_size = cfg.training.batch_size
        for start in range(0, len(progress), batch_size):
            indices = progress[start:start + batch_size]
            batch = [train_queries[i] for i in indices]
            batch_passages = list(dict.fromkeys(rng.choice(q.relevant_document_ids) for q in batch))
            mask = positive_mask(batch, batch_passages).to(model.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                query_vectors = model.encode(train_tokens[indices])
                passage_vectors = model.encode(
                    passage_tokens[[row_by_id[pid] for pid in batch_passages]], passage=True,
                )
                loss = loss_function(query_vectors, passage_vectors, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.detach().item() * len(batch)
        model.eval()
        mrr = development_mrr(model, dev_queries, dev_tokens, passage_tokens, document_ids, cfg.batch_size)
        entry = {"epoch": epoch, "train_loss": total_loss / len(train_queries), "dev_mrr_at_10": mrr}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if mrr > best:
            best = mrr
            experiment = OmegaConf.to_container(cfg, resolve=True)
            experiment["selection"] = entry
            experiment["dataset"] = cfg.dataset.name
            experiment["training_dtype"] = "bfloat16" if use_bfloat16 else "float32"
            experiment["torch_version"] = str(torch.__version__)
            experiment["checkpoint_id"] = uuid4().hex
            save_checkpoint(model, checkpoint, experiment)
    (checkpoint.parent / "training.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode()
def development_mrr(model, queries, query_tokens, passage_tokens, document_ids, batch_size):
    corpus_vectors = model.encode_batches(passage_tokens, batch_size, passage=True)
    relevancies = []
    start = 0
    for token_batch in query_tokens.split(batch_size):
        scores = model.encode(token_batch) @ corpus_vectors.T
        indices = scores.topk(min(10, len(document_ids)), dim=1).indices.cpu().tolist()
        for query, ranking in zip(queries[start:start + len(indices)], indices, strict=True):
            relevant = set(query.relevant_document_ids)
            relevancies.append([document_ids[i] in relevant for i in ranking])
        start += len(indices)
    return Evaluate(relevancies, queries, (10,))[10]["mrr"]


def save_checkpoint(model, path, experiment):
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save({
            "model": model.model_config,
            "tokenizer": model.tokenizer.to_str(),
            "weights": model.state_dict(),
            "experiment": experiment,
        }, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    model.tokenizer.no_padding()
    model.tokenizer.no_truncation()
    model.tokenizer.save(str(path.parent / "tokenizer.json"))
    (path.parent / "configuration.json").write_text(json.dumps(experiment, indent=2) + "\n", encoding="utf-8")
