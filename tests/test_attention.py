import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data_utils.document import Document
from src.data_utils.query import Query
from src.evaluation.attention import get_attention, save_examples, select_examples
from src.retrievers.dpr import DPR, fit_tokenizer


def tiny_model(pooling="mean"):
    torch.manual_seed(17)
    tokenizer = fit_tokenizer(["bond prices rise"], ["interest rates fall"], 64)
    return DPR(tokenizer, dict(d_model=8, n_heads=2, n_layers=2, feed_forward_dim=16,
                               pooling_type=pooling, dropout=.5, vocab_size=64,
                               query_length=12, passage_length=16))


@pytest.mark.parametrize("pooling", ["cls", "mean", "max"])
def test_attention_is_unpadded_and_uses_eval_mode(pooling):
    model = tiny_model(pooling)
    first = get_attention(model, "bond prices")
    second = get_attention(model, "bond prices")
    assert model.training  # Extraction restores the caller's mode.
    np.testing.assert_array_equal(first["attention"], second["attention"])
    assert "[PAD]" not in first["tokens"]
    length = len(first["tokens"])
    assert first["attention"].shape == (2, 2, length, length)
    np.testing.assert_allclose(first["attention"].sum(-1), 1, atol=1e-6)
    assert not first["truncated"]
    assert get_attention(model, "bond " * 30, passage=True)["truncated"]


def test_selected_outcomes_ranks_and_saved_arrays(tmp_path):
    documents = [Document("bond prices rise", "a"), Document("interest rates fall", "b")]
    retriever = SimpleNamespace(model=tiny_model().eval(), documents=documents,
                                Lookup=lambda queries, k: [documents[:k] for _ in queries])
    queries = [Query("bond prices", "hit", ["a"]), Query("interest rates", "miss", ["b"])]
    examples = select_examples(retriever, queries, k=1, count=1)
    assert [(e["outcome"], e["relevant_rank"]) for e in examples] == [("hit", 1), ("miss", 2)]
    output = save_examples(retriever.model, examples, tmp_path / "attention")
    saved = json.loads((output / "examples.json").read_text())
    assert saved[0]["inputs"]["query"]["truncated"] is False
    with np.load(output / "01_hit/query.npz", allow_pickle=False) as arrays:
        expected = get_attention(retriever.model, queries[0].text)
        np.testing.assert_array_equal(arrays["attention"], expected["attention"])
        assert arrays["tokens"].tolist() == expected["tokens"]
    assert not (output / "01_hit/top_passage.npz").exists()  # Same as judged passage.
    assert (output / "02_miss/top_passage.npz").exists()
    assert (output / "run.json").exists()
    with pytest.raises(FileExistsError):
        save_examples(retriever.model, examples, output)


def test_individual_head_panels_preserve_weights():
    from matplotlib import pyplot as plt
    from src.evaluation.attention import plot_attention

    data = get_attention(tiny_model(), "bond prices")
    fig, axes = plt.subplots(1, 2)
    try:
        for head, ax in enumerate(axes, 1):
            plot_attention(data, layer=2, head=head, ax=ax)
            np.testing.assert_array_equal(ax.images[0].get_array(), data["attention"][1, head - 1])
    finally:
        plt.close(fig)


def test_plot_command_exports_individual_heads(tmp_path):
    import subprocess
    import sys

    data = get_attention(tiny_model(), "bond prices")
    array = tmp_path / "query.npz"
    np.savez_compressed(array, attention=data["attention"], tokens=np.array(data["tokens"]))
    output = tmp_path / "heads"
    command = [
        sys.executable, "-m", "src.evaluation.plot_attention",
        "--panel", str(array), "1", "1",
        "--panel", str(array), "2", "2",
        "--output", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert output.with_suffix(".pdf").stat().st_size > 0
    assert output.with_suffix(".png").stat().st_size > 0

    invalid_output = tmp_path / "invalid"
    result = subprocess.run([
        sys.executable, "-m", "src.evaluation.plot_attention",
        "--panel", str(array), "1", "0", "--output", str(invalid_output),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Choose a head" in result.stderr
    assert not invalid_output.with_suffix(".pdf").exists()
