from src.data_utils.query import Query


def calculateRecall(relevancies: list[bool], query: Query):
    total = query.getNumberOfRelevantDocuments()
    return sum(relevancies) / total if total else 0.0


def Evaluate(relevancies: list[list[bool]], queries: list[Query], cutoffs=(1, 5, 10, 20)):
    if len(relevancies) != len(queries) or not queries:
        raise ValueError("Evaluation requires one ranking per query and at least one query")
    scores = {}
    for k in cutoffs:
        if k <= 0:
            raise ValueError("Cutoffs must be positive")
        precision = recall = reciprocal_rank = 0.0
        for ranking, query in zip(relevancies, queries):
            top = ranking[:k]
            precision += sum(top) / k
            recall += calculateRecall(top, query)
            reciprocal_rank += next((1 / rank for rank, hit in enumerate(top, 1) if hit), 0)
        scores[k] = {
            "precision": precision / len(queries),
            "recall": recall / len(queries),
            "mrr": reciprocal_rank / len(queries),
        }
    return scores
