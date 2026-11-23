import hashlib
import random
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset, DatasetDict, load_from_disk

from src.data_utils.download_retrieval_datasets import get_fiqa_dataset, get_squad_dataset
from src.data_utils.query import Query

SPLITS = ("train", "dev", "test")


class Corpus:
    def __init__(self, dataset_name: str, base_dir: str, seed: int = 42):
        if dataset_name not in {"fiqa", "squad"}:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.dataset_name = dataset_name
        root = Path(base_dir)
        cache = root / "prepared" / (f"squad-{seed}" if dataset_name == "squad" else "fiqa")
        if cache.exists():
            self.ds = load_from_disk(str(cache))
            self._validate()
            return

        raw = str(root / "raw")
        self.ds = self._prepare_fiqa(raw) if dataset_name == "fiqa" else self._prepare_squad(raw, seed)
        self._validate()
        cache.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache.parent) as staging:
            path = Path(staging) / "prepared"
            self.ds.save_to_disk(str(path))
            try:
                path.rename(cache)
            except OSError:
                if not cache.exists():
                    raise

    @property
    def train(self):
        return self.ds["train"]

    @property
    def val(self):
        return self.ds["dev"]

    @property
    def test(self):
        return self.ds["test"]

    @property
    def all_documents(self):
        return self.ds["corpus"]

    def queries(self, split):
        return [Query(row["question"], row["query_id"], row["document_id"]) for row in self.ds[split]]

    def _validate(self):
        document_ids = set(self.all_documents["document_id"])
        if len(document_ids) != len(self.all_documents) or not document_ids:
            raise ValueError("Corpus must have unique passage IDs and at least one passage")
        self.relevant_ids = {}
        seen_articles = set()
        for split in SPLITS:
            if not len(self.ds[split]):
                raise ValueError(f"Empty query split: {split}")
            for row in self.ds[split]:
                query_id = row["query_id"]
                relevant = row["document_id"]
                if query_id in self.relevant_ids or not relevant or not document_ids.issuperset(relevant):
                    raise ValueError(f"Invalid query ID or relevance judgments: {query_id}")
                self.relevant_ids[query_id] = relevant
            if self.dataset_name == "squad":
                articles = set(self.ds[split]["article"])
                if seen_articles & articles:
                    raise ValueError(f"SQuAD articles overlap in split {split}")
                seen_articles |= articles

    @staticmethod
    def _prepare_fiqa(raw):
        corpus, queries, judgments = get_fiqa_dataset(raw)
        texts = {str(row["_id"]): row["text"] for row in queries}
        result = {
            "corpus": Dataset.from_dict({
                "document_id": [str(value) for value in corpus["_id"]],
                "text": corpus["text"],
            }),
        }
        for split, positives in judgments.items():
            result[split] = Dataset.from_list([
                {"query_id": query_id, "question": texts[query_id], "document_id": list(dict.fromkeys(ids))}
                for query_id, ids in positives.items()
            ])
        return DatasetDict(result)

    @staticmethod
    def _prepare_squad(raw, seed):
        squad = get_squad_dataset(raw)
        articles = sorted(set(squad["train"]["title"]))
        development = set(random.Random(seed).sample(articles, round(len(articles) * 0.1)))
        documents = {}
        rows = {split: [] for split in SPLITS}
        for source in ("train", "validation"):
            for row in squad[source]:
                article = row["title"]
                if source == "validation":
                    split = "test"
                else:
                    split = "dev" if article in development else "train"
                text = row["context"]
                if text not in documents:
                    documents[text] = hashlib.sha256(text.encode("utf-8")).hexdigest()
                rows[split].append({
                    "query_id": row["id"],
                    "question": row["question"],
                    "document_id": [documents[text]],
                    "article": article,
                })
        result = {split: Dataset.from_list(queries) for split, queries in rows.items()}
        result["corpus"] = Dataset.from_dict({"text": list(documents), "document_id": list(documents.values())})
        return DatasetDict(result)
