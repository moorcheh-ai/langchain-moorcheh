# Import all the necessary files and packages
import asyncio
import inspect
import json
import logging
import os
import re
import time
from typing import Any, List, Literal, Optional, Sequence, Tuple, Type, TypeVar
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from moorcheh_sdk import APIError, MoorchehClient  # type: ignore

# Set up functions for logging
logger = logging.getLogger(__name__)

# Define parameters
NamespaceType = Literal["text", "vector"]
VST = TypeVar("VST", bound=VectorStore)


class MoorchehVectorStore(VectorStore):
    def __init__(
        self,
        api_key: str,
        namespace: str,
        namespace_type: NamespaceType = "text",
        vector_dimension: Optional[int] = None,  # Required for vector namespace
        embedding: Optional[Embeddings] = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        if not api_key:
            api_key = os.getenv("MOORCHEH_API_KEY") or ""
        if not api_key:
            raise ValueError(
                "`api_key` is required for Moorcheh client initialization. "
                "Provide it directly or set the MOORCHEH_API_KEY environment variable."
            )

        if namespace_type == "vector" and vector_dimension is None:
            raise ValueError(
                "For 'vector' namespace_type, 'vector_dimension' must be provided."
            )
        if namespace_type not in ["text", "vector"]:
            raise ValueError(
                f"Invalid 'namespace_type': {namespace_type}. "
                "Must be 'text' or 'vector'."
            )

        self._client = MoorchehClient(api_key=api_key)
        self.namespace = namespace
        self.namespace_type = namespace_type
        self.vector_dimension = vector_dimension
        self.embedding = embedding
        self.batch_size = batch_size

        try:
            namespaces_response = self._list_namespaces()
            namespaces_names = [
                ns["namespace_name"] for ns in namespaces_response.get("namespaces", [])
            ]
            logger.info("Found namespaces.")
        except Exception as e:
            logger.error(f"Failed to list namespaces: {e}")
            raise

        # If namespace exists, add to existing namespace. If not found, creates it.
        if self.namespace in namespaces_names:
            logger.info(
                f"Namespace '{self.namespace}' already exists. No action required."
            )
        else:
            logger.info(f"Namespace '{self.namespace}' not found. Creating it.")
            try:
                self._create_namespace(
                    namespace_name=self.namespace,
                    type=self.namespace_type,
                    vector_dimension=self.vector_dimension,
                )
            except Exception as e:
                logger.error(f"Failed to create namespace: {e}")
                raise

    # Embeddings property
    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    # Class Method: From texts
    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VST:
        # LangChain Document object
        documents = []

        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts.")
        if ids and len(ids) != len(texts):
            raise ValueError("Length of ids must match length of texts.")

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))

        # Create a configured store instance (kwargs should include api_key/namespace)
        # Remove embedding from kwargs since parent VectorStore doesn't accept it
        instance = cls(**kwargs)
        instance.embedding = embedding  # type: ignore

        # Upload the constructed documents to Moorcheh; if ids is None,
        # IDs will be derived/generated
        instance.add_documents(documents=documents, ids=ids)

        return instance

    # method: add documents
    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs: Any
    ) -> List[str]:
        # if documents aren't provided, return empty
        if not documents:
            return []

        # if namespace type vector, use vector uploads.
        if self.namespace_type != "text":
            raise ValueError(
                "add_documents is only valid for 'text' namespaces. "
                "Use upload_vectors for 'vector'."
            )

        # if ids are provided, and they don't match length of documents -
        # raise value error.
        if ids is not None and len(ids) != len(documents):
            raise ValueError(
                "Number of IDs must match number of documents if provided."
            )

        moorcheh_docs_to_upload: List[dict] = []
        assigned_ids: List[str] = []

        # adds document id for each document
        for i, doc in enumerate(documents):
            if ids is not None:
                doc_id = str(ids[i])
            else:
                doc_id = (
                    # if doc id isn't provided by user then uses a random uuid.
                    str(getattr(doc, "id"))
                    if getattr(doc, "id", None) is not None
                    else (
                        str(getattr(doc, "id_"))
                        if getattr(doc, "id_", None) is not None
                        else str(uuid4())
                    )
                )

            metadata = (doc.metadata or {}).copy()

            # prepares documents_to_upload with id, text, and metadata
            moorcheh_doc = {
                "id": doc_id,
                "text": doc.page_content,
                "metadata": metadata,
            }
            moorcheh_docs_to_upload.append(moorcheh_doc)
            assigned_ids.append(doc_id)

        # uploads the documents to moorcheh sdk
        self._upload_documents(
            namespace_name=self.namespace,
            documents=moorcheh_docs_to_upload,
        )

        return assigned_ids

    # Method: Upload Vectors
    def upload_vectors(
        self,
        vectors: List[Tuple[str, List[float], Optional[dict]]],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        # if vectors aren't provided, return empty
        if not vectors:
            return []

        # if namespace type isn't vector, raise value error
        if self.namespace_type != "vector":
            raise ValueError("upload_vectors is only valid for 'vector' namespaces.")

        # if ids are provided, and they don't match length of vectors -
        # raise value error.
        if ids is not None and len(ids) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors if provided.")

        moorcheh_vector_to_upload: List[dict] = []
        assigned_ids: List[str] = []

        # adds document id for each vector
        for i, (vector_id, vector, metadata) in enumerate(vectors):
            if ids is not None:
                vector_id = str(ids[i])
            else:
                vector_id = (
                    # if vector id isn't provided by user then uses a random uuid.
                    vector_id.strip()
                    if isinstance(vector_id, str) and vector_id.strip()
                    else str(uuid4())
                )

            meta = (metadata or {}).copy()

            # prepares vectors_to_upload with id, vector, and metadata
            moorcheh_vec = {
                "id": vector_id,
                "vector": vector,
                "metadata": meta,
            }
            moorcheh_vector_to_upload.append(moorcheh_vec)
            assigned_ids.append(vector_id)

        # uploads the vectors to moorcheh sdk
        self._upload_vectors(
            namespace_name=self.namespace,
            vectors=moorcheh_vector_to_upload,
        )

        return assigned_ids

    # Delete method
    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        # if no ID is provided, cannot delete anything.
        if not ids:
            return False

        try:
            # if namespace type is text, delete documents.
            if self.namespace_type == "text":
                logger.info(
                    f"Deleting {len(ids)} documents from Moorcheh (text namespace)..."
                )
                self._delete_documents(namespace_name=self.namespace, ids=ids)
            # if namespace type is vector, delete vectors.
            elif self.namespace_type == "vector":
                logger.info(
                    f"Deleting {len(ids)} vectors from Moorcheh (vector namespace)..."
                )
                self._delete_vectors(namespace_name=self.namespace, ids=ids)
            # if any other type, raise value error.
            else:
                raise ValueError(f"Unsupported namespace type: {self.namespace_type}")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def delete_namespace(self) -> bool:
        """Delete the entire namespace and all its contents.

        This method is useful for cleanup in tests and when you want to
        completely remove a namespace.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(
                f"Deleting namespace '{self.namespace}' and all its contents..."
            )

            try:
                self._delete_namespace(namespace_name=self.namespace)
                logger.info(f"Successfully deleted namespace '{self.namespace}'")
                return True
            except Exception as e:
                logger.error(f"Failed to delete namespace '{self.namespace}': {e}")
                return False

        except Exception as e:
            logger.error(f"Error in delete_namespace: {e}")
            return False

    # Similarity_search method
    def similarity_search(
        self,
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            # if vector, then the query provided must be embedded.
            if self.namespace_type == "vector":
                if isinstance(query, str):
                    raise ValueError(
                        "In a 'vector' namespace, query must be an embedded "
                        "vector (not text)."
                    )

            # Call SDK for search
            search_results = self._search_with_retries(query=query, k=k, **kwargs)

            # Obtains results
            results = search_results.get("results", []) or []

            # Organizes results
            documents: List[Document] = []
            for result in results:
                page_content = result.get("text")

                # Normalize id
                doc_id = result.get("id")
                doc_id = str(doc_id) if doc_id is not None else None

                # Metadata from API + keep label if present
                raw_metadata = result.get("metadata") or {}
                metadata = (
                    raw_metadata.get("metadata", raw_metadata)
                    if isinstance(raw_metadata, dict)
                    else {}
                )

                # Builds the Langchain document
                documents.append(
                    Document(page_content=page_content, metadata=metadata, id=doc_id)
                )

            return documents

        except Exception as e:
            logger.error(f"Error executing similarity search: {e}")
            raise

    # Similarity_search with score method
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        try:
            # if vector, then the query provided must be embedded.
            if self.namespace_type == "vector":
                if isinstance(query, str):
                    raise ValueError(
                        "In a 'vector' namespace, query must be an embedded "
                        "vector (not text)."
                    )

            # Call SDK for search
            search_results = self._search_with_retries(query=query, k=k, **kwargs)

            # Obtains results
            results = search_results.get("results", []) or []

            # Organizes results
            scored_langchain_docs: List[Tuple[Document, float]] = []
            for result in results:
                # score extraction
                try:
                    score = float(result.get("score", 0.0))
                except Exception:
                    score = 0.0

                # get all the content
                page_content = result.get("text")

                doc_id = result.get("id")
                doc_id = str(doc_id) if doc_id is not None else None

                raw_metadata = result.get("metadata") or {}
                metadata = (
                    raw_metadata.get("metadata", raw_metadata)
                    if isinstance(raw_metadata, dict)
                    else {}
                )

                # Builds the Langchain document + pair with score
                scored_langchain_docs.append(
                    (
                        Document(
                            page_content=page_content, metadata=metadata, id=doc_id
                        ),
                        score,
                    )
                )

            return scored_langchain_docs
        except Exception as e:
            logger.error(f"Error executing similarity search with score: {e}")
            raise

    # Generative_answer Method: Only for a text namespace
    def generative_answer(self, query: str, k: int = 10, **kwargs: Any) -> str:
        try:
            # Only valid for text namespaces
            if self.namespace_type != "text":
                raise ValueError(
                    "generative_answer is only valid for 'text' namespaces."
                )

            # Call SDK for get generative answer endpoint
            result = self._generate_answer(
                namespace=self.namespace,
                query=query,
                top_k=k,
                # can specify additional parameters, such as ai_model, temperature, etc.
                **kwargs,
            )
            return result.get("answer", "")
        except Exception as e:
            logger.error(f"Error getting generative answer: {e}")
            raise

    # Get_by_ids method: Only for text namespace
    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        # if no ids specified, return empty
        if not ids:
            return []

        # Only valid for text namespaces
        if self.namespace_type != "text":
            raise ValueError("get_by_ids is only valid for 'text' namespaces.")

        try:
            # Call sdk to get documents
            response = self._get_documents(
                namespace_name=self.namespace,
                ids=ids,
            )

            items = response.get("items") or []
            docs: List[Document] = []

            for item in items:
                # Obtain content
                text = item.get("text") or ""
                raw_metadata = item.get("metadata") or {}
                metadata = (
                    raw_metadata.get("metadata", raw_metadata)
                    if isinstance(raw_metadata, dict)
                    else {}
                )

                # Normalize id
                doc_id = item.get("id")
                doc_id = str(doc_id) if doc_id is not None else None

                # Append to LangChain Document list
                docs.append(Document(page_content=text, metadata=metadata, id=doc_id))

            # Preserve input order
            by_id = {doc.id: doc for doc in docs if doc.id is not None}
            return [by_id[str(i)] for i in ids if str(i) in by_id]
        except APIError as e:
            # Check if this is a Status 207 (partial success) case
            if "207" in str(e) or "partial" in str(e).lower():
                # Status 207 means partial success - try to extract partial results
                try:
                    logger.warning(f"Status 207 received (partial success): {e}")

                    # Try to extract partial results from the error response
                    # The error might contain the partial response data
                    error_str = str(e)

                    # Try to find JSON content in the error message
                    json_match = re.search(r"\{.*\}", error_str, re.DOTALL)
                    if json_match:
                        try:
                            partial_response = json.loads(json_match.group())
                            items = partial_response.get("items", [])

                            if items:
                                partial_docs: List[Document] = []
                                for item in items:
                                    # Obtain content
                                    text = item.get("text") or ""
                                    raw_metadata = item.get("metadata") or {}
                                    metadata = (
                                        raw_metadata.get("metadata", raw_metadata)
                                        if isinstance(raw_metadata, dict)
                                        else {}
                                    )

                                    # Normalize id
                                    doc_id = item.get("id")
                                    doc_id = str(doc_id) if doc_id is not None else None

                                    # Append to LangChain Document list
                                    partial_docs.append(
                                        Document(
                                            page_content=text,
                                            metadata=metadata,
                                            id=doc_id,
                                        )
                                    )

                            # Preserve input order for found documents
                            by_id = {
                                doc.id: doc
                                for doc in partial_docs
                                if doc.id is not None
                            }
                            return [by_id[str(i)] for i in ids if str(i) in by_id]
                        except (json.JSONDecodeError, KeyError, TypeError):
                            logger.warning(
                                "Could not parse partial response from Status 207"
                            )

                    # If we can't extract partial results, return empty list
                    return []

                except Exception as parse_error:
                    logger.warning(
                        f"Could not parse Status 207 response: {parse_error}"
                    )
                    return []
            elif "Unexpected response format from get documents endpoint" in str(e):
                # This might also be a partial success case
                logger.warning(f"Unexpected response format error: {e}")
                return []
            # For other API errors, re-raise
            logger.error(f"APIError in get_by_ids: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_by_ids: {e}")
            raise

    """ Async Methods """

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        if not ids:
            return False

        if self.namespace_type == "text":
            logger.info(
                f"Deleting {len(ids)} documents from Moorcheh (text namespace)..."
            )
            await asyncio.to_thread(
                self._delete_documents,
                namespace_name=self.namespace,
                ids=ids,
            )
        elif self.namespace_type == "vector":
            logger.info(
                f"Deleting {len(ids)} vectors from Moorcheh (vector namespace)..."
            )
            await asyncio.to_thread(
                self._delete_vectors,
                namespace_name=self.namespace,
                ids=ids,
            )
        else:
            raise ValueError(f"Unsupported namespace type: {self.namespace_type}")
        return True

    @classmethod
    async def afrom_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VST:
        documents: List[Document] = []
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts.")
        if ids and len(ids) != len(texts):
            raise ValueError("Length of ids must match length of texts.")

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))

        # Remove embedding from kwargs since parent VectorStore doesn't accept it
        instance = cls(**kwargs)
        instance.embedding = embedding  # type: ignore
        await instance.aadd_documents(documents=documents, ids=ids)
        return instance

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if not documents:
            return []

        # Only valid for text namespaces
        if self.namespace_type != "text":
            raise ValueError(
                "add_documents is only valid for 'text' namespaces. "
                "Use upload_vectors for 'vector'."
            )

        if ids is not None and len(ids) != len(documents):
            raise ValueError(
                "Number of IDs must match number of documents if provided."
            )

        moorcheh_docs_to_upload: List[dict] = []
        assigned_ids: List[str] = []

        for i, doc in enumerate(documents):
            if ids is not None:
                doc_id = str(ids[i])
            else:
                doc_id = (
                    str(getattr(doc, "id"))
                    if getattr(doc, "id", None) is not None
                    else (
                        str(getattr(doc, "id_", None))
                        if getattr(doc, "id_", None) is not None
                        else str(uuid4())
                    )
                )

            metadata = (doc.metadata or {}).copy()

            moorcheh_docs_to_upload.append(
                {"id": doc_id, "text": doc.page_content, "metadata": metadata}
            )
            assigned_ids.append(doc_id)

        await asyncio.to_thread(
            self._upload_documents,
            namespace_name=self.namespace,
            documents=moorcheh_docs_to_upload,
        )
        return assigned_ids

    async def aupload_vectors(
        self,
        vectors: List[Tuple[str, List[float], Optional[dict]]],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if not vectors:
            return []

        # Only valid for vector namespaces
        if self.namespace_type != "vector":
            raise ValueError("upload_vectors is only valid for 'vector' namespaces.")

        if ids is not None and len(ids) != len(vectors):
            raise ValueError("Number of IDs must match number of vectors if provided.")

        moorcheh_vector_to_upload: List[dict] = []
        assigned_ids: List[str] = []

        for i, (vector_id, vector, metadata) in enumerate(vectors):
            if ids is not None:
                vector_id = str(ids[i])
            else:
                vector_id = (
                    vector_id.strip()
                    if isinstance(vector_id, str) and vector_id.strip()
                    else str(uuid4())
                )

            meta = (metadata or {}).copy()

            moorcheh_vec = {
                "id": vector_id,
                "vector": vector,
                "metadata": meta,
            }
            moorcheh_vector_to_upload.append(moorcheh_vec)
            assigned_ids.append(vector_id)

        await asyncio.to_thread(
            self._upload_vectors,
            namespace_name=self.namespace,
            vectors=moorcheh_vector_to_upload,
        )
        return assigned_ids

    async def asimilarity_search(
        self,
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> List[Document]:
        if self.namespace_type == "vector" and isinstance(query, str):
            raise ValueError(
                "In a 'vector' namespace, query must be an embedded vector (not text)."
            )

        search_results = await self._async_search_with_retries(
            query=query, k=k, **kwargs
        )

        results = search_results.get("results", []) or []
        documents: List[Document] = []

        for result in results:
            page_content = result.get("text")

            # Normalize id
            doc_id = result.get("id")
            doc_id = str(doc_id) if doc_id is not None else None

            # metadata
            raw_metadata = result.get("metadata", {})
            metadata = raw_metadata.get("metadata", raw_metadata)

            documents.append(
                Document(page_content=page_content, metadata=metadata, id=doc_id)
            )

        return documents

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        # same namespace guard as sync
        if self.namespace_type == "vector" and isinstance(query, str):
            raise ValueError(
                "In a 'vector' namespace, query must be an embedded vector (not text)."
            )

        search_results = await self._async_search_with_retries(
            query=query, k=k, **kwargs
        )

        results = search_results.get("results", []) or []
        scored_langchain_docs: List[Tuple[Document, float]] = []

        for result in results:
            # score extraction
            try:
                score = float(result.get("score", 0.0))
            except Exception:
                score = 0.0

            page_content = result.get("text")

            # Normalize id
            doc_id = result.get("id")
            doc_id = str(doc_id) if doc_id is not None else None

            raw_metadata = result.get("metadata", {})
            metadata = raw_metadata.get("metadata", raw_metadata)

            scored_langchain_docs.append(
                (
                    Document(page_content=page_content, metadata=metadata, id=doc_id),
                    score,
                )
            )

        return scored_langchain_docs

    async def agenerative_answer(self, query: str, k: int = 10, **kwargs: Any) -> str:
        # text namespaces only
        if self.namespace_type != "text":
            raise ValueError("generative_answer is only valid for 'text' namespaces.")

        result = await asyncio.to_thread(
            self._generate_answer,
            namespace=self.namespace,
            query=query,
            top_k=k,
            # allow ai_model/temperature, other parameters, etc.
            **kwargs,
        )
        return result.get("answer", "")

    async def _async_search_with_retries(
        self, query: str, k: int, **kwargs: Any
    ) -> dict:
        """Retry async search briefly to tolerate indexing propagation delay."""
        retries = int(kwargs.pop("consistency_retries", 3))
        retry_delay = float(kwargs.pop("consistency_retry_delay", 0.5))

        # Vector namespaces typically don't need indexing retries.
        if self.namespace_type == "vector":
            retries = 1

        search_results: dict = {}
        for attempt in range(max(1, retries)):
            search_results = await asyncio.to_thread(
                self._search,
                namespaces=[self.namespace],
                query=query,
                top_k=k,
                **kwargs,
            )
            if attempt < retries - 1:
                await asyncio.sleep(retry_delay)
        return search_results

    async def aget_by_ids(self, ids: Sequence[str]) -> List[Document]:
        if not ids:
            return []

        if self.namespace_type != "text":
            raise ValueError("get_by_ids is only valid for 'text' namespaces.")

        try:
            response = await asyncio.to_thread(
                self._get_documents,
                namespace_name=self.namespace,
                ids=ids,
            )
        except APIError as e:
            # Check if this is a Status 207 (partial success) case
            if "207" in str(e) or "partial" in str(e).lower():
                # Status 207 means partial success - try to extract partial results
                try:
                    logger.warning(f"Status 207 received (partial success): {e}")

                    # Try to extract partial results from the error response
                    # The error might contain the partial response data
                    error_str = str(e)

                    # Try to find JSON content in the error message
                    json_match = re.search(r"\{.*\}", error_str, re.DOTALL)
                    if json_match:
                        try:
                            partial_response = json.loads(json_match.group())
                            items = partial_response.get("items", [])

                            if items:
                                async_docs: List[Document] = []
                                for item in items:
                                    text = item.get("text") or ""

                                    raw_metadata = item.get("metadata") or {}
                                    metadata = (
                                        raw_metadata.get("metadata", raw_metadata)
                                        if isinstance(raw_metadata, dict)
                                        else {}
                                    )

                                    # id normalization
                                    doc_id = item.get("id")
                                    doc_id = str(doc_id) if doc_id is not None else None

                                    async_docs.append(
                                        Document(
                                            page_content=text,
                                            metadata=metadata,
                                            id=doc_id,
                                        )
                                    )

                                # preserve input order for found documents
                                by_id = {
                                    doc.id: doc
                                    for doc in async_docs
                                    if doc.id is not None
                                }
                                return [by_id[str(i)] for i in ids if str(i) in by_id]
                        except (json.JSONDecodeError, KeyError, TypeError):
                            logger.warning(
                                "Could not parse partial response from Status 207"
                            )

                    # If we can't extract partial results, return empty list
                    return []

                except Exception as parse_error:
                    logger.warning(
                        f"Could not parse Status 207 response: {parse_error}"
                    )
                    return []
            elif "Unexpected response format from get documents endpoint" in str(e):
                # This might also be a partial success case
                logger.warning(f"Unexpected response format error: {e}")
                return []
            # For other API errors, re-raise
            raise

        items = response.get("items") or []
        docs: List[Document] = []

        for item in items:
            text = item.get("text") or ""

            raw_metadata = item.get("metadata") or {}
            metadata = (
                raw_metadata.get("metadata", raw_metadata)
                if isinstance(raw_metadata, dict)
                else {}
            )

            # id normalization
            doc_id = item.get("id")
            doc_id = str(doc_id) if doc_id is not None else None

            docs.append(Document(page_content=text, metadata=metadata, id=doc_id))

        # preserve input order
        by_id = {doc.id: doc for doc in docs if doc.id is not None}
        return [by_id[str(i)] for i in ids if str(i) in by_id]

    def _resolve_api(self, *path: str) -> Optional[Any]:
        current: Any = self._client
        for part in path:
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current

    def _call_with_supported_kwargs(self, fn: Any, **kwargs: Any) -> Any:
        try:
            signature = inspect.signature(fn)
            if any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            ):
                return fn(**kwargs)
            accepted = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
            return fn(**accepted)
        except (ValueError, TypeError):
            return fn(**kwargs)

    def _list_namespaces(self) -> dict:
        namespaced = self._resolve_api("namespaces", "list")
        if callable(namespaced):
            return self._call_with_supported_kwargs(namespaced)
        raise AttributeError("Missing required SDK API: namespaces.list")

    def _create_namespace(self, **kwargs: Any) -> Any:
        namespaced = self._resolve_api("namespaces", "create")
        if callable(namespaced):
            return self._call_with_supported_kwargs(namespaced, **kwargs)
        raise AttributeError("Missing required SDK API: namespaces.create")

    def _upload_documents(self, **kwargs: Any) -> Any:
        namespaced = self._resolve_api("documents", "upload")
        if callable(namespaced):
            return self._call_with_supported_kwargs(namespaced, **kwargs)
        raise AttributeError("Missing required SDK API: documents.upload")

    def _upload_vectors(self, **kwargs: Any) -> Any:
        namespaced = self._resolve_api("vectors", "upload")
        if callable(namespaced):
            return self._call_with_supported_kwargs(namespaced, **kwargs)
        raise AttributeError("Missing required SDK API: vectors.upload")

    def _delete_documents(self, **kwargs: Any) -> Any:
        namespaced = self._resolve_api("documents", "delete")
        if callable(namespaced):
            return self._call_with_supported_kwargs(namespaced, **kwargs)
        raise AttributeError("Missing required SDK API: documents.delete")

    def _delete_vectors(self, **kwargs: Any) -> Any:
        namespaced = self._resolve_api("vectors", "delete")
        if callable(namespaced):
            return self._call_with_supported_kwargs(namespaced, **kwargs)
        raise AttributeError("Missing required SDK API: vectors.delete")

    def _delete_namespace(self, namespace_name: str) -> Any:
        namespaced = self._resolve_api("namespaces", "delete")
        if callable(namespaced):
            try:
                return self._call_with_supported_kwargs(
                    namespaced, namespace_name=namespace_name
                )
            except TypeError:
                return namespaced(namespace_name)
        raise AttributeError("Missing required SDK API: namespaces.delete")

    def _search(self, **kwargs: Any) -> Any:
        namespaced = self._resolve_api("similarity_search", "query")
        if callable(namespaced):
            return self._call_with_supported_kwargs(namespaced, **kwargs)
        raise AttributeError("Missing required SDK API: similarity_search.query")

    def _search_with_retries(self, query: str, k: int, **kwargs: Any) -> dict:
        retries = int(kwargs.pop("consistency_retries", 3))
        retry_delay = float(kwargs.pop("consistency_retry_delay", 0.5))

        if self.namespace_type == "vector":
            retries = 1

        search_results: dict = {}
        for attempt in range(max(1, retries)):
            search_results = self._search(
                namespaces=[self.namespace],
                query=query,
                top_k=k,
                **kwargs,
            )
            if attempt < retries - 1:
                time.sleep(retry_delay)
        return search_results

    def _generate_answer(self, **kwargs: Any) -> Any:
        namespaced = self._resolve_api("answer", "generate")
        if callable(namespaced):
            return self._call_with_supported_kwargs(namespaced, **kwargs)
        raise AttributeError("Missing required SDK API: answer.generate")

    def _get_documents(self, **kwargs: Any) -> Any:
        namespaced = self._resolve_api("documents", "get")
        if callable(namespaced):
            return self._call_with_supported_kwargs(namespaced, **kwargs)
        raise AttributeError("Missing required SDK API: documents.get")
