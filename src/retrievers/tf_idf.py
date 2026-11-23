import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from src.retrievers.base_retriever import Retriever


WORDS = re.compile(r"[^\W_]+")


def tokenize(text):
    return WORDS.findall(text.lower())


class TFIDF(Retriever):
    def __init__(self, documents=None, index_path=None):
        super().__init__(documents, index_path)
        counts = [Counter(tokenize(doc.GetText())) for doc in self.index.GetDocuments()]
        self.document_frequencies = Counter(term for row in counts for term in row)
        self.term_to_index = {term: i for i, term in enumerate(sorted(self.document_frequencies))}
        self.idf = self.GetInverseDocumentFrequencies()
        self.matrix = self._matrix(counts, document_weights=True)

    def GetInverseDocumentFrequencies(self):
        size = len(self.index.GetDocuments())
        return {term: np.log((size + 1) / (df + 1)) + 1 for term, df in self.document_frequencies.items()}

    def _term_weight(self, term, count, length):
        return count * self.idf[term]

    def _matrix(self, counts, document_weights=False):
        data, columns, offsets = [], [], [0]
        for row in counts:
            length = sum(row.values())
            for term, count in row.items():
                if term in self.term_to_index:
                    columns.append(self.term_to_index[term])
                    data.append(self._term_weight(term, count, length) if document_weights else count)
            offsets.append(len(columns))
        return csr_matrix((data, columns, offsets), shape=(len(offsets) - 1, len(self.term_to_index)), dtype=np.float32)

    def _scores(self, queries):
        query_counts = (Counter(tokenize(query)) for query in queries)
        return self._matrix(query_counts) @ self.matrix.T

    def CalculateScores(self, queries):
        return self._scores(queries).toarray()

    def Lookup(self, queries, k):
        products = self._scores(queries).tocsr()
        documents = self.index.GetDocuments()
        k = max(0, min(k, len(documents)))
        rankings = []
        for row in range(products.shape[0]):
            start, end = products.indptr[row:row + 2]
            ids = products.indices[start:end]
            scores = products.data[start:end]
            order = np.lexsort((ids, -scores))
            selected = ids[order[:k]].tolist()
            if len(selected) < k:
                scored = set(selected)
                for index in range(len(documents)):
                    if index not in scored:
                        selected.append(index)
                    if len(selected) == k:
                        break
            rankings.append([documents[index] for index in selected])
        return rankings
