import json
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np

from src.data_utils.dataset import DocumentDataset


class VectorIndex:
    def __init__(self, index, passage_ids, checkpoint_id):
        if index.ntotal != len(passage_ids) or len(passage_ids) != len(set(passage_ids)):
            raise ValueError("The FAISS index requires one unique ID per passage vector")
        self.index = index
        self.passage_ids = passage_ids
        self.checkpoint_id = checkpoint_id

    @classmethod
    def build(cls, vectors, passage_ids, checkpoint_id):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, passage_ids, checkpoint_id)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Keep each checkpoint's index and mapping together.
        target = directory / self.checkpoint_id
        with TemporaryDirectory(dir=directory) as staging:
            bundle = Path(staging) / "index"
            bundle.mkdir()
            faiss.write_index(self.index, str(bundle / "index.faiss"))
            (bundle / "passages.json").write_text(json.dumps({
                "passage_ids": self.passage_ids,
                "checkpoint_id": self.checkpoint_id,
            }), encoding="utf-8")
            try:
                bundle.rename(target)
            except OSError:
                if not target.is_dir():
                    raise

    @classmethod
    def load(cls, directory, checkpoint_id):
        directory = Path(directory) / checkpoint_id
        metadata = json.loads((directory / "passages.json").read_text(encoding="utf-8"))
        if metadata["checkpoint_id"] != checkpoint_id:
            raise ValueError("The FAISS index belongs to a different DPR checkpoint")
        index = faiss.read_index(str(directory / "index.faiss"))
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("Expected a FAISS IndexFlatIP")
        return cls(index, metadata["passage_ids"], checkpoint_id)

    def search(self, query_vectors, k):
        k = min(k, len(self.passage_ids))
        if k <= 0:
            return np.empty((len(query_vectors), 0), dtype=np.int64)
        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
        return self.index.search(queries, k)[1]


class DenseRetriever:
    def __init__(self, model, documents, index, batch_size):
        self.model = model
        self.documents = DocumentDataset(documents).GetDocuments()
        self.index = index
        self.batch_size = batch_size
        if [doc.GetId() for doc in self.documents] != index.passage_ids:
            raise ValueError("FAISS passage IDs differ from the prepared corpus")
        if index.index.d != model.query_encoder.d_model:
            raise ValueError("FAISS vector width differs from the query encoder")

    def Lookup(self, queries, k):
        if not queries:
            return []
        tokens = self.model.tokenize(queries)
        vectors = self.model.encode_batches(tokens, self.batch_size).float().cpu().numpy()
        return [[self.documents[i] for i in row] for row in self.index.search(vectors, k)]


def load_dense_retriever(cfg, corpus):
    from src.retrievers.dpr import DPR
    from src.training.trainer import checkpoint_path

    checkpoint = checkpoint_path(cfg)
    model, experiment = DPR.from_checkpoint(checkpoint, cfg.device)
    if experiment["dataset"] != corpus.dataset_name or experiment["seed"] != cfg.seed:
        raise ValueError("DPR checkpoint dataset or seed differs from this run")
    checkpoint_id = experiment["checkpoint_id"]
    path = checkpoint.parent / "indexes"
    documents = corpus.all_documents
    if not (path / checkpoint_id).is_dir():
        tokens = model.tokenize(documents["text"], passage=True)
        vectors = model.encode_batches(tokens, cfg.batch_size, passage=True).float().cpu().numpy()
        VectorIndex.build(vectors, documents["document_id"], checkpoint_id).save(path)
    index = VectorIndex.load(path, checkpoint_id)
    return DenseRetriever(model, documents, index, cfg.batch_size), experiment
