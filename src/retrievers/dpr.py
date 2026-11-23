from itertools import chain

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from torch import nn

from src.retrievers.models.encoder import TransformerEncoder


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]


def fit_tokenizer(queries, passages, vocab_size):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.train_from_iterator(
        chain(queries, passages),
        trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, show_progress=False),
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[(token, tokenizer.token_to_id(token)) for token in ("[CLS]", "[SEP]")],
    )
    return tokenizer


def tokenize_batch(tokenizer, texts, max_length):
    tokenizer.enable_truncation(max_length)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), length=max_length)
    return torch.tensor([encoded.ids for encoded in tokenizer.encode_batch(texts)], dtype=torch.long)


class DPR(nn.Module):
    def __init__(self, tokenizer, model_config):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_config = dict(model_config)
        options = dict(self.model_config)
        self.query_length = options.pop("query_length")
        self.passage_length = options.pop("passage_length")
        options.pop("vocab_size")
        options.update(
            vocab_size=tokenizer.get_vocab_size(),
            max_len=max(self.query_length, self.passage_length),
            padding_idx=tokenizer.token_to_id("[PAD]"),
        )
        self.query_encoder = TransformerEncoder(**options)
        self.passage_encoder = TransformerEncoder(**options)

    @property
    def device(self):
        return next(self.parameters()).device

    def tokenize(self, texts, *, passage=False):
        if not texts:
            return torch.empty((0, self.passage_length if passage else self.query_length), dtype=torch.long)
        length = self.passage_length if passage else self.query_length
        return torch.cat([tokenize_batch(self.tokenizer, texts[i:i + 1024], length) for i in range(0, len(texts), 1024)])

    def encode(self, tokens, *, passage=False):
        encoder = self.passage_encoder if passage else self.query_encoder
        # Trim padding shared by the entire batch before transferring to the accelerator.
        width = int((tokens != encoder.padding_idx).sum(dim=1).max())
        return encoder(tokens[:, :width].to(self.device))

    @torch.inference_mode()
    def encode_batches(self, tokens, batch_size, *, passage=False):
        return torch.cat([self.encode(batch, passage=passage) for batch in tokens.split(batch_size)])

    @classmethod
    def from_checkpoint(cls, path, device):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(Tokenizer.from_str(state["tokenizer"]), state["model"])
        model.load_state_dict(state["weights"])
        return model.to(device).eval(), state["experiment"]
