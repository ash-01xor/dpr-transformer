"""Choose retrieval examples, extract token attention, and draw heatmaps."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


@torch.inference_mode()
def get_attention(model, text, passage=False):
    """Read one input's attention: [layer, head, receiving token, source token]."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        encoder = model.passage_encoder if passage else model.query_encoder
        token_ids = model.tokenize([text], passage=passage)
        token_ids = token_ids[:, token_ids[0] != encoder.padding_idx]
        _, layers = encoder(token_ids.to(model.device), return_attention=True)

        # Count the original tokens to tell us whether the model lost any text.
        tokenizer = Tokenizer.from_str(model.tokenizer.to_str())
        tokenizer.no_padding()
        tokenizer.no_truncation()
        original_length = len(tokenizer.encode(text).ids)
        return {
            "tokens": [model.tokenizer.id_to_token(int(i)) for i in token_ids[0]],
            "token_ids": token_ids[0].tolist(),
            "attention": torch.stack([layer[0].float().cpu() for layer in layers]).numpy(),
            "truncated": original_length > token_ids.shape[1],
        }
    finally:
        for module, training in modes:
            module.training = training


def plot_attention(data, layer=1, head=None, max_tokens=64, vmax=1.0, ax=None):
    """Draw one layer. Layers/heads start at 1; head=None averages the heads."""
    from matplotlib import pyplot as plt

    weights = data["attention"]
    if not 1 <= layer <= len(weights):
        raise ValueError(f"Choose a layer from 1 to {len(weights)}")
    if head is not None and not 1 <= head <= weights.shape[1]:
        raise ValueError(f"Choose a head from 1 to {weights.shape[1]}, or None")
    if max_tokens < 1 or not 0 < vmax <= 1:
        raise ValueError("max_tokens must be positive and vmax must be in (0, 1]")
    matrix = weights[layer - 1].mean(axis=0) if head is None else weights[layer - 1, head - 1]
    size = min(max_tokens, len(data["tokens"]))
    labels = [f"{i}: {token}" for i, token in enumerate(data["tokens"][:size])]
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(max(8, size * .20), max(7, size * .18)))
    else:
        fig = ax.figure
    image = ax.imshow(matrix[:size, :size], cmap="magma", vmin=0, vmax=vmax)
    ax.set_xticks(range(size), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(size), labels, fontsize=7)
    ax.set_xlabel("Source token (information read from)")
    ax.set_ylabel("Receiving token (representation updated)")
    title = f"Layer {layer}, " + ("mean of heads" if head is None else f"head {head}")
    if size < len(data["tokens"]):
        title += f"\nShowing first {size}/{len(data['tokens'])} tokens; weights are not renormalized"
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Attention weight",
                 extend="max" if matrix[:size, :size].max() > vmax else "neither")
    if own_figure:
        fig.tight_layout()
    return fig


def select_examples(retriever, queries, k=10, count=2, batch_size=64):
    """Keep the first count hits and misses in query order, from one split.

    A hit has a judged relevant passage in the top k. A miss has none there.
    An unjudged passage is not necessarily irrelevant.
    """
    if min(k, count, batch_size) < 1:
        raise ValueError("k, count and batch_size must be positive")
    examples = []
    counts = {"hit": 0, "miss": 0}
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        rankings = retriever.Lookup([q.text for q in batch], k)
        for query, ranking in zip(batch, rankings, strict=True):
            relevant_ids = set(query.relevant_document_ids)
            outcome = "hit" if any(doc.GetId() in relevant_ids for doc in ranking) else "miss"
            if counts[outcome] >= count:
                continue
            # For selected examples, get the exact rank of the best judged passage.
            full = retriever.Lookup([query.text], len(retriever.documents))[0]
            judged = next(((rank, doc) for rank, doc in enumerate(full, 1)
                           if doc.GetId() in relevant_ids), None)
            if not full or judged is None:
                raise ValueError(f"No indexed judged passage for query {query.id}")
            rank, document = judged
            examples.append({
                "query_id": str(query.id), "query": query.text, "outcome": outcome,
                "relevant_ids": list(query.relevant_document_ids), "relevant_rank": rank,
                "judged_passage": {"id": document.GetId(), "text": document.GetText()},
                "top_passage": {"id": full[0].GetId(), "text": full[0].GetText()},
                "top_k_ids": [doc.GetId() for doc in full[:k]],
            })
            counts[outcome] += 1
        if all(number >= count for number in counts.values()):
            break
    print(f"Selected {counts['hit']} hits and {counts['miss']} misses at {k} (requested {count} each).")
    return examples


def save_examples(model, examples, output, run_info=None):
    """Save case texts/ranks, full arrays, and early/late mean-head heatmaps."""
    from matplotlib import pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    saved = []
    for number, example in enumerate(examples, 1):
        directory = output / f"{number:02d}_{example['outcome']}"
        directory.mkdir()
        inputs = [("query", example["query"], False),
                  ("judged_passage", example["judged_passage"]["text"], True)]
        if example["top_passage"]["id"] != example["judged_passage"]["id"]:
            inputs.append(("top_passage", example["top_passage"]["text"], True))
        saved_example = dict(example, folder=directory.name, inputs={})
        for name, text, passage in inputs:
            data = get_attention(model, text, passage=passage)
            np.savez_compressed(directory / f"{name}.npz", attention=data["attention"],
                                tokens=np.array(data["tokens"]), token_ids=np.array(data["token_ids"]))
            saved_example["inputs"][name] = {"tokens": len(data["tokens"]), "truncated": data["truncated"]}
            for layer in sorted({1, len(data["attention"])}):
                fig = plot_attention(data, layer=layer)
                fig.savefig(directory / f"{name}_layer{layer}.png", dpi=120)
                plt.close(fig)
        saved.append(saved_example)
    (output / "examples.json").write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    info = dict(run_info or {}, model=model.model_config)
    (output / "run.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved examples and attention maps to {output}")
    return output


def attention_report(cfg, corpus, retriever, experiment):
    if cfg.split not in {"dev", "test"}:
        raise ValueError("Choose split=dev or split=test for attention analysis")
    examples = select_examples(retriever, corpus.queries(cfg.split), k=cfg.attention.k,
                               count=cfg.attention.examples_per_outcome, batch_size=cfg.batch_size)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(cfg.attention.output_root) / cfg.dataset.name / timestamp
    return save_examples(retriever.model, examples, output, run_info={
        "dataset": cfg.dataset.name, "split": cfg.split, "k": cfg.attention.k,
        "selection": "first examples of each outcome in dataset query order",
        "checkpoint": str(Path(cfg.output_dir) / cfg.dataset.name / "dpr/checkpoint.pt"),
        "experiment": experiment,
    })
