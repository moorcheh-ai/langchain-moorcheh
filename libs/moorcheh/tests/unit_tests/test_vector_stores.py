import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing_extensions import Literal

from langchain_moorcheh import MoorchehVectorStore

MOORCHEH_API_KEY = "test-api-key"
OPENAI_API_KEY = "test-openai-key"
NAMESPACE_NAME = "test-namespace"
NAMESPACE_TYPE: Literal["text", "vector"] = "text"


class TestMoorchehVectorStore(unittest.TestCase):
    @patch("langchain_moorcheh.vectorstores.MoorchehClient")
    def setUp(self, mock_client_class: Any) -> None:
        self.mock_client = MagicMock()
        mock_client_class.return_value = self.mock_client
        self.mock_client.namespaces = MagicMock()
        self.mock_client.namespaces.list.return_value = {"namespaces": []}
        self.mock_client.namespaces.create = MagicMock()
        self.mock_client.namespaces.delete = MagicMock()

        self.mock_client.documents = MagicMock()
        self.mock_client.documents.upload = MagicMock()
        self.mock_client.documents.get = MagicMock()
        self.mock_client.documents.delete = MagicMock()

        self.mock_client.vectors = MagicMock()
        self.mock_client.vectors.upload = MagicMock()
        self.mock_client.vectors.delete = MagicMock()

        self.mock_client.similarity_search = MagicMock()
        self.mock_client.similarity_search.query = MagicMock()

        self.mock_client.answer = MagicMock()
        self.mock_client.answer.generate = MagicMock()

        self.embedding = MagicMock(spec=OpenAIEmbeddings)
        self.embedding.embed_query.return_value = [0.1] * 1536

        self.store = MoorchehVectorStore(
            api_key=MOORCHEH_API_KEY,
            namespace=NAMESPACE_NAME,
            namespace_type=NAMESPACE_TYPE,
            embedding=self.embedding,
        )

    def test_add_documents(self) -> None:
        documents = [Document(page_content="Test content", metadata={"source": "unit"})]
        added_ids = self.store.add_documents(documents=documents)

        self.assertEqual(len(added_ids), 1)
        self.store._client.documents.upload.assert_called_once()

    def test_similarity_search(self) -> None:
        mock_results = {
            "results": [
                {"text": "Mocked result", "metadata": {"source": "mock"}},
            ]
        }
        self.store._client.similarity_search.query.return_value = mock_results

        results = self.store.similarity_search("mock query", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Mocked result")
        self.assertEqual(results[0].metadata["source"], "mock")

    def test_similarity_search_with_score(self) -> None:
        mock_results = {
            "results": [
                {
                    "text": "Mocked result",
                    "metadata": {"source": "mock"},
                    "score": 0.88,
                }
            ]
        }
        self.store._client.similarity_search.query.return_value = mock_results

        results = self.store.similarity_search_with_score("mock query", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].page_content, "Mocked result")
        self.assertAlmostEqual(results[0][1], 0.88, places=2)

    def test_get_by_ids(self) -> None:
        self.store._client.documents.get.return_value = {
            "items": [{"id": "123", "text": "Doc by ID", "metadata": {"id": "123"}}]
        }

        # Corrected line: remove the namespace_name keyword argument
        results = self.store.get_by_ids(ids=["123"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Doc by ID")
        self.assertEqual(results[0].metadata["id"], "123")

    def test_delete_documents(self) -> None:
        success = self.store.delete(ids=["id1", "id2"])
        self.assertTrue(success)
        self.store._client.documents.delete.assert_called_once()

    def test_delete_vectors(self) -> None:
        self.store.namespace_type = "vector"
        success = self.store.delete(ids=["v1", "v2"])
        self.assertTrue(success)
        self.store._client.vectors.delete.assert_called_once()

    def test_upload_vectors(self) -> None:
        self.store.namespace_type = "vector"
        vectors = [("id1", [0.1] * 1536, {"tag": "test"})]
        returned_ids = self.store.upload_vectors(vectors)  # type: ignore[arg-type]

        self.assertEqual(returned_ids, ["id1"])
        self.store._client.vectors.upload.assert_called_once()

    def test_from_texts(self) -> None:
        with patch(
            "langchain_moorcheh.vectorstores.MoorchehClient",
            return_value=self.mock_client,
        ):
            self.mock_client.namespaces.list.return_value = {"namespaces": []}

            texts = ["Doc 1", "Doc 2"]
            metadatas = [{"source": "a"}, {"source": "b"}]

            store = MoorchehVectorStore.from_texts(
                texts=texts,
                embedding=self.embedding,
                metadatas=metadatas,
                api_key=MOORCHEH_API_KEY,
                namespace=NAMESPACE_NAME,
                namespace_type=NAMESPACE_TYPE,
            )

            self.assertIsInstance(store, MoorchehVectorStore)
            self.mock_client.documents.upload.assert_called_once()

    def test_generative_answer(self) -> None:
        self.store._client.answer.generate.return_value = {
            "answer": "This is an answer."
        }
        result = self.store.generative_answer("test query", k=1)
        self.assertEqual(result, "This is an answer.")

    def test_add_documents_with_ids(self) -> None:
        docs = [Document(page_content="text", metadata={})]
        ids = ["custom-id"]

        returned_ids = self.store.add_documents(docs, ids=ids)
        self.assertEqual(returned_ids, ids)
        self.store._client.documents.upload.assert_called_once()

    def test_delete_no_ids(self) -> None:
        result = self.store.delete(ids=None)
        self.assertFalse(result)

    def test_delete_namespace(self) -> None:
        """Test the delete_namespace method."""
        self.store._client.namespaces.delete.return_value = None

        result = self.store.delete_namespace()
        self.assertTrue(result)
        self.store._client.namespaces.delete.assert_called_once_with(
            namespace_name=self.store.namespace
        )


if __name__ == "__main__":
    unittest.main()
