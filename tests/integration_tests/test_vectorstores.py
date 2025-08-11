from typing import Generator
import os
import uuid
import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests
from langchain_moorcheh import MoorchehVectorStore

MOORCHEH_API_KEY = "test-api-key"
NAMESPACE_NAME = "test-integration-ns"
NAMESPACE_TYPE = "text"

class TestMoorchehVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty Moorcheh vectorstore for integration tests."""
        if not MOORCHEH_API_KEY:
            pytest.skip("MOORCHEH_API_KEY not set. Skipping integration test.")
        
        # Generate a unique namespace name for this test run
        unique_namespace_name = f"{NAMESPACE_NAME}-{uuid.uuid4()}"

        store = MoorchehVectorStore(
            api_key=MOORCHEH_API_KEY,
            namespace=unique_namespace_name,
            namespace_type=NAMESPACE_TYPE,
            embedding=self.get_embeddings(),
        )
        try:
            yield store
        finally:
            try:
                store.delete()
            except Exception:
                pass
'''

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_add_documents_async(self): ...

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_vectorstore_is_empty_async(self): ...

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_vectorstore_still_empty_async(self): ...

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_deleting_documents_async(self): ...

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_deleting_bulk_documents_async(self): ...

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_delete_missing_content_async(self): ...

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_add_documents_with_ids_is_idempotent_async(self): ...

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_add_documents_by_id_with_mutation_async(self): ...

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_get_by_ids_async(self): ...

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_get_by_ids_missing_async(self): ...

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_add_documents_documents_async(self): ...

    @pytest.mark.xfail(reason="MoorchehVectorStore does not support async methods.")
    def test_add_documents_with_existing_ids_async(self): ...

'''