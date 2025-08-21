import uuid
from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_moorcheh import MoorchehVectorStore

MOORCHEH_API_KEY = "test-api-key" #Place your moorcheh API key
NAMESPACE_NAME = "test-integration-ns"
NAMESPACE_TYPE = "text"


class TestMoorchehVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:   
        """Get an empty Moorcheh vectorstore for integration tests."""
        if not MOORCHEH_API_KEY:
            pytest.skip("MOORCHEH_API_KEY not set. Skipping integration test.")
        
        # Generate a unique namespace name for each test run
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
