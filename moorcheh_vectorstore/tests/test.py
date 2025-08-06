import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from moorcheh_langchain import MoorchehVectorStore

MOORCHEH_API_KEY = "test-api-key"
OPENAI_API_KEY = "test-openai-key"
NAMESPACE_NAME = "test-namespace"
NAMESPACE_TYPE = "text"

class TestMoorchehVectorStore(unittest.TestCase):

    @patch("moorcheh_langchain.MoorchehClient")
    def setUp(self, mock_client_class):
        self.mock_client = MagicMock()
        mock_client_class.return_value = self.mock_client
        self.mock_client.list_namespaces.return_value = {"namespaces": []}

        self.embedding = MagicMock(spec=OpenAIEmbeddings)
        self.embedding.embed_query.return_value = [0.1] * 1536

        self.store = MoorchehVectorStore(
            api_key=MOORCHEH_API_KEY,
            namespace=NAMESPACE_NAME,
            namespace_type=NAMESPACE_TYPE,
            embedding=self.embedding,
        )

    def test_add_documents(self):
        documents = [Document(page_content="Test content", metadata={"source": "unit"})]
        self.store._client.upload_documents = MagicMock()
        added_ids = self.store.add_documents(documents=documents)

        self.assertEqual(len(added_ids), 1)
        self.store._client.upload_documents.assert_called_once()

    def test_similarity_search(self):
        mock_results = {
            "results": [
                {"text": "Mocked result", "metadata": {"source": "mock"}},
            ]
        }
        self.store._client.search.return_value = mock_results

        results = self.store.similarity_search("mock query", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Mocked result")
        self.assertEqual(results[0].metadata["source"], "mock")

    def test_similarity_search_with_score(self):
        mock_results = {
            "results": [
                {
                    "text_content": "Mocked result",
                    "metadata": {"source": "mock"},
                    "score": 0.88,
                }
            ]
        }
        self.store._client.search.return_value = mock_results

        results = self.store.similarity_search_with_score("mock query", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].page_content, "Mocked result")
        self.assertAlmostEqual(results[0][1], 0.88, places=2)
