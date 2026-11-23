import torch

from src.data_utils.query import Query
from src.retrievers.dpr import DPR, fit_tokenizer
from src.training.loss import DPRLoss
from src.training.trainer import positive_mask


def test_multi_positive_loss_and_tiny_overfit():
    torch.manual_seed(42)
    query_texts = ["apples and oranges", "rain and thunder", "blue whale", "music and piano"]
    passages = ["orange orchard", "heavy thunderstorm", "whales in the ocean", "piano concerto"]
    ids = ["a", "b", "c", "d"]
    queries = [Query(text, identifier, [identifier]) for text, identifier in zip(query_texts, ids)]
    mask = positive_mask(queries, ids)
    queries[0].relevant_document_ids.append("b")
    assert positive_mask(queries, ids)[0].tolist() == [True, True, False, False]
    criterion = DPRLoss()
    assert criterion(torch.zeros(1, 2), torch.zeros(2, 2), torch.ones(1, 2, dtype=torch.bool)) == 0

    settings = {
        "d_model": 16, "n_heads": 2, "n_layers": 1, "feed_forward_dim": 32,
        "pooling_type": "cls", "dropout": 0, "vocab_size": 96, "query_length": 12, "passage_length": 12,
    }
    tokenizer = fit_tokenizer(query_texts, passages, settings["vocab_size"])
    model = DPR(tokenizer, settings)
    query_tokens = model.tokenize(query_texts)
    passage_tokens = model.tokenize(passages, passage=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    def vectors():
        return model.encode(query_tokens), model.encode(passage_tokens, passage=True)

    def hits():
        with torch.no_grad():
            query_vectors, passage_vectors = vectors()
            return int(((query_vectors @ passage_vectors.T).argmax(dim=1) == torch.arange(len(ids))).sum())

    initial_hits = hits()
    for step in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(*vectors(), mask)
        loss.backward()
        if step == 0:
            for encoder in (model.query_encoder, model.passage_encoder):
                assert all(p.grad is not None and p.grad.isfinite().all() for p in encoder.parameters())
                assert any(p.grad.abs().sum() > 0 for p in encoder.parameters())
        optimizer.step()
    assert hits() == len(ids) > initial_hits
