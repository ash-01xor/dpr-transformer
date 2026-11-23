from src.data_utils.document import Document


class DocumentDataset:
    def __init__(self, documents) -> None:
        self.documents = [Document(text=row["text"], _id=row["document_id"]) for row in documents]

    def GetDocuments(self) -> list[Document]:
        return self.documents
