import os
import uuid
from typing import Any

import pytest
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests
from typing_extensions import Literal

from langchain_moorcheh import MoorchehVectorStore

MOORCHEH_API_KEY = os.getenv("MOORCHEH_API_KEY", "test-api-key")
NAMESPACE_NAME = "test-integration-ns"
NAMESPACE_TYPE: Literal["text", "vector"] = "text"


@pytest.mark.compile
class TestMoorchehVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Any:
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
                # Use delete_namespace for proper cleanup
                store.delete_namespace()
            except Exception:
                pass
