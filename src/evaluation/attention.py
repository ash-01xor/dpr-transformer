from html import escape
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.retrievers.tf_idf import tokenize


def select_examples(retriever, corpus, batch_size):
    examples = {}
    for split in ("test", "dev"):
        for query, ranking in ranked_queries(retriever, corpus.queries(split), batch_size):
            relevant = set(query.relevant_document_ids)
            hit = next((doc for doc in ranking if doc.GetId() in relevant), None)
            outcome = "hit" if hit else "miss"
            if outcome in examples:
                continue
            if hit:
                reference = hit
                rank = ranking.index(hit) + 1
            else:
                # Retrieve the best judged passage and its exact rank for the missed example.
                full_ranking = retriever.Lookup([query.text], len(retriever.documents))[0]
                rank, reference = next(
                    (rank, doc) for rank, doc in enumerate(full_ranking, 1) if doc.GetId() in relevant
                )
            examples[outcome] = (split, query, reference, rank, ranking[0])
            if len(examples) == 2:
                return examples
    if not examples:
        raise ValueError("The attention report requires ranked queries")
    return examples


def ranked_queries(retriever, queries, batch_size):
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        rankings = retriever.Lookup([query.text for query in batch], 10)
        yield from zip(batch, rankings, strict=True)


@torch.inference_mode()
def plot_attention(model, text, path, *, passage=False):
    encoder = model.passage_encoder if passage else model.query_encoder
    token_ids = model.tokenize([text], passage=passage)
    token_ids = token_ids[:, token_ids[0] != encoder.padding_idx]
    tokens = [model.tokenizer.id_to_token(int(token)) for token in token_ids[0]]
    _, layers = encoder(token_ids.to(model.device), return_attention=True)
    layer_numbers = sorted({0, len(layers) - 1})
    observations = {}
    with PdfPages(path) as pdf:
        for layer in layer_numbers:
            weights = layers[layer][0].float().cpu().numpy()
            heads = len(weights)
            size = max(9, min(25, 5 + 0.15 * len(tokens)))
            fig, axes = plt.subplots((heads + 1) // 2, 2, figsize=(size * 1.5, size), squeeze=False)
            v_max = float(weights.max())
            fontsize = max(4, min(8, 240 / len(tokens)))
            observations[layer + 1] = []
            for head, (ax, matrix) in enumerate(zip(axes.flat, weights), 1):
                image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=v_max, interpolation="nearest")
                ax.set_title(f"Head {head}")
                ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=fontsize)
                ax.set_yticks(range(len(tokens)), tokens, fontsize=fontsize)
                ax.tick_params(length=0)
                ax.set_xlabel("Key token")
                ax.set_ylabel("Query token")
                peaks = np.argsort(-matrix[0], kind="stable")[:3]
                observations[layer + 1].append([(tokens[i], float(matrix[0, i])) for i in peaks])
            for ax in list(axes.flat)[heads:]:
                ax.set_visible(False)
            fig.suptitle(f"{'Passage' if passage else 'Query'} encoder, layer {layer + 1}")
            fig.subplots_adjust(left=0.09, right=0.88, top=0.91, bottom=0.10, wspace=0.35, hspace=0.50)
            fig.colorbar(image, ax=list(axes.flat)[:heads], label="Attention weight", shrink=0.7)
            pdf.savefig(fig)
            plt.close(fig)
    return observations


def excerpt(text, length=360):
    text = " ".join(text.split())
    return text if len(text) <= length else text[:length].rsplit(" ", 1)[0] + "…"


def shared_terms(query, passage):
    terms = set(tokenize(passage))
    return list(dict.fromkeys(word for word in tokenize(query) if len(word) > 3 and word in terms))


def describe_attention(observations):
    early, late = list(observations.values())[0], list(observations.values())[-1]
    early_peaks = [head[0][0] for head in early]
    late_peaks = [head[0][0] for head in late]
    changed = sum(a != b for a, b in zip(early_peaks, late_peaks, strict=True))
    early_weight = sum(head[0][1] for head in early) / len(early)
    late_weight = sum(head[0][1] for head in late) / len(late)
    return (
        f"Across heads, the highest-weight key for `[CLS]` changes in {changed}/{len(early)} heads. "
        f"The leading keys cover {len(set(early_peaks))} distinct tokens in the early layer and "
        f"{len(set(late_peaks))} in the late layer. "
        f"The mean leading weight moves from {early_weight:.1%} to {late_weight:.1%}."
    )


def attention_report(cfg, corpus, retriever):
    if cfg.split != "test":
        raise ValueError("Use split=test for the held-out attention report")
    output = Path(cfg.output_dir) / cfg.dataset.name / "dpr" / "attention"
    output.mkdir(parents=True, exist_ok=True)
    examples = select_examples(retriever, corpus, cfg.batch_size)
    lines = [
        f"# {cfg.dataset.name.upper()} DPR attention",
        "",
        "A hit places a judged passage in the top 10. A miss puts the closest judged passage below rank 10.",
        "",
    ]
    for outcome in ("hit", "miss"):
        if outcome not in examples:
            lines.extend([f"No {outcome} at 10 occurred in either development or test.", ""])
            continue
        split, query, reference, rank, top = examples[outcome]
        judged_terms = shared_terms(query.text, reference.GetText())
        first_terms = shared_terms(query.text, top.GetText())
        lines.extend([
            f"## {outcome.title()} at 10 · {split}",
            "",
            f"Query {query.id} · {query.text}",
            "",
            f"- Judged passage {reference.GetId()} at rank {rank} · {excerpt(reference.GetText())}",
            f"- First result {top.GetId()} · {excerpt(top.GetText())}",
            f"- Shared query terms with the judged passage · {', '.join(judged_terms) or 'none'}",
            f"- Shared query terms with the first result · {', '.join(first_terms) or 'none'}",
            "",
        ])
        if rank == 1:
            lines.append("The highest-scoring pooled passage vector belongs to a judged passage.")
        else:
            lines.append(
                f"DPR placed the first result above the nearest judged passage. The first result shares "
                f"{len(first_terms)} query terms of four or more characters, and the judged passage "
                f"shares {len(judged_terms)}."
            )
        lines.append("")
        inputs = [("query", query.text, False), ("judged passage", reference.GetText(), True)]
        if top.GetId() != reference.GetId():
            inputs.append(("first result", top.GetText(), True))
        for label, text, passage in inputs:
            name = f"{outcome}_{label.replace(' ', '_')}.pdf"
            peak_weights = plot_attention(retriever.model, text, output / name, passage=passage)
            lines.extend([
                f"### {label.capitalize()}",
                "",
                f"[Per-head attention heatmaps]({name})",
                "",
                describe_attention(peak_weights),
                "",
                "| Layer | Head | Top three keys for the CLS token |",
                "| --- | --- | --- |",
            ])
            for layer, heads in peak_weights.items():
                for head, peaks in enumerate(heads, 1):
                    keys = ", ".join(
                        f"<code>{escape(token).replace('|', '&#124;')}</code> ({weight:.1%})"
                        for token, weight in peaks
                    )
                    lines.append(f"| {layer} | {head} | {keys} |")
            lines.append("")
    lines.extend([
        "The two encoders apply attention within their own token sequences. Passage order comes from "
        "the dot product between their pooled CLS vectors.",
        "",
    ])
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)
