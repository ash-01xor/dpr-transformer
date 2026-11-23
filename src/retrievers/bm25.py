import numpy as np

from src.retrievers.tf_idf import TFIDF, tokenize


class BM25(TFIDF):
    def __init__(self, documents=None, index_path=None, k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        super().__init__(documents, index_path)

    def GetInverseDocumentFrequencies(self):
        size = len(self.index.GetDocuments())
        return {term: np.log1p((size - df + 0.5) / (df + 0.5)) for term, df in self.document_frequencies.items()}

    def GetDocumentLengths(self):
        lengths = {doc: len(tokenize(doc.GetText())) for doc in self.index.GetDocuments()}
        return lengths, sum(lengths.values()) / max(len(lengths), 1)

    def _matrix(self, counts, document_weights=False):
        if document_weights:
            self.average_document_length = sum(sum(row.values()) for row in counts) / max(len(counts), 1)
        return super()._matrix(counts, document_weights)

    def _term_weight(self, term, count, length):
        normalization = self.k1 * (1 - self.b + self.b * length / self.average_document_length)
        return self.idf[term] * count * (self.k1 + 1) / (count + normalization)

    @property
    def bm25_matrix(self):
        return self.matrix
