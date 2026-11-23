def retrieveQueryAndGetRelevancies(model, queries, k, batch_size=128):
    relevancies = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        rankings = model.Lookup([query.getQuery() for query in batch], k)
        for query, documents in zip(batch, rankings, strict=True):
            relevant = set(query.relevant_document_ids)
            seen = set()
            hits = []
            for document in documents:
                document_id = document.GetId()
                hits.append(document_id in relevant and document_id not in seen)
                seen.add(document_id)
            relevancies.append(hits)
    return relevancies
