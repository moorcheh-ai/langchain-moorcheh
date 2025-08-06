from langchain.vectorstores import VectorStore
from moorcheh_sdk import MoorchehClient

import logging
import os
import uuid
from typing import Any, List, Optional, Literal, Tuple, Type, TypeVar, Sequence
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

# Use Literal for namespace type
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
            api_key = os.getenv("MOORCHEH_API_KEY")
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
                f"Invalid 'namespace_type': {namespace_type}. Must be 'text' or 'vector'."
            )

        self._client = MoorchehClient(api_key=api_key)
        self.namespace = namespace
        self.namespace_type = namespace_type
        self.vector_dimension = vector_dimension
        self.embedding = embedding
        self.batch_size = batch_size

        try:
            namespaces_response = self._client.list_namespaces()
            namespaces_names = [
                ns["namespace_name"]
                for ns in namespaces_response.get("namespaces", [])
            ]
            print("Found namespaces.")
        except Exception as e:
            print(f"Failed to list namespaces: {e}")
            raise

        if self.namespace in namespaces_names:
            print(f"Namespace '{self.namespace}' already exists. No action required.")
        else:
            print(f"Namespace '{self.namespace}' not found. Creating it.")
            try:
                self._client.create_namespace(
                    namespace_name=self.namespace,
                    type=self.namespace_type,
                    vector_dimension=self.vector_dimension,
                )
            except Exception as e:
                print(f"Failed to create namespace: {e}")
                raise

    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        if not ids:
            return False

        try:
            if self.namespace_type == "text":
                print(f"Deleting {len(ids)} documents from Moorcheh (text namespace)...")
                self._client.delete_documents(namespace_name=self.namespace, ids=ids)
            elif self.namespace_type == "vector":
                print(f"Deleting {len(ids)} vectors from Moorcheh (vector namespace)...")
                self._client.delete_vectors(namespace_name=self.namespace, ids=ids)
            else:
                raise ValueError(f"Unsupported namespace type: {self.namespace_type}")
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            raise

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> VST:
        documents = []
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts.")
        if ids and len(ids) != len(texts):
            raise ValueError("Length of ids must match length of texts.")

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            doc_id = ids[i] if ids else str(uuid.uuid4())
            documents.append(Document(page_content=text, metadata=metadata))

        instance = cls(embedding=embedding, **kwargs)
        instance.add_documents(documents=documents, ids=ids)
        return instance

    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs: Any
    ) -> List[str]:
        if not documents:
            return []

        if ids is not None and len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents if provided.")

        moorcheh_docs_to_upload = []
        assigned_ids = []

        for i, doc in enumerate(documents):
            doc_id = ids[i] if ids else str(uuid.uuid4())
            assigned_ids.append(doc_id)

            moorcheh_doc = {
                "id": doc_id,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            moorcheh_docs_to_upload.append(moorcheh_doc)

        self._client.upload_documents(
            namespace_name=self.namespace,
            documents=moorcheh_docs_to_upload,
        )
        return assigned_ids

    def upload_vectors(
        self,
        vectors: List[Tuple[str, List[float], Optional[dict]]],  # (id, vector, metadata)
    ) -> List[str]:
        if self.namespace_type != "vector":
            raise ValueError("upload_vectors is only valid for 'vector' namespaces.")

        payload = []
        ids = []

        for doc_id, vector, metadata in vectors:
            payload.append({
                "id": doc_id,
                "vector": vector,
                "metadata": metadata or {},
            })
            ids.append(doc_id)

        self._client.upload_vectors(
            namespace_name=self.namespace,
            vectors=payload,
        )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            search_results = self._client.search(
                namespaces=[self.namespace],
                query=query,
                top_k=k,
                **kwargs
            )

            documents: List[Document] = []

            for result in search_results.get("results", []):
                page_content = result.get("text") or result.get("text_content", f"[Vector ID: {result.get('id', '')}]")
                metadata = result.get("metadata", {})

                documents.append(Document(page_content=page_content, metadata=metadata))

            return documents

        except Exception as e:
            logger.error(f"Error executing similarity search: {e}")
            raise


    def similarity_search_with_score(
      self, query: str, k: int = 4, **kwargs: Any
  ) -> List[Tuple[Document, float]]:
      try:
          search_results = self._client.search(
              namespaces=[self.namespace],
              query=query,
              top_k=k,
              **kwargs
          )

          results = search_results.get("results", [])
          scored_langchain_docs: List[Tuple[Document, float]] = []

          for result in results:
              score = result.get("score", 0.0)
              try:
                  score = float(score)
              except Exception:
                  score = 0.0

              page_content = result.get("text") or result.get("text_content", "")
              metadata = result.get("metadata", {})

              scored_langchain_docs.append(
                  (Document(page_content=page_content, metadata=metadata), score)
              )

          return scored_langchain_docs

      except Exception as e:
          logger.error(f"Error executing similarity search with score: {e}")
          raise



    def generative_answer(self, query: str, k: int = 4, **kwargs):
        try:
            result = self._client.get_generative_answer(
                namespace=self.namespace,
                query=query,
                top_k=k,
                ai_model = "anthropic.claude-3-7-sonnet-20250219-v1:0",
                **kwargs,
            )
            return result.get("answer", "")
        except Exception as e:
            logger.error(f"Error getting generative answer: {e}")
            raise

    def get_by_ids(self, namespace_name: str, ids: List[str]):
      try:
            response = self._client.get_documents(
                namespace_name=self.namespace,
                ids=ids
            )
            documents_data = response.get("documents", [])

            langchain_documents = []
            for doc_data in documents_data:
                page_content = doc_data.get("text")
                metadata = doc_data.get("metadata", {})

                if page_content:
                    langchain_documents.append(
                        Document(page_content=page_content, metadata=metadata)
                    )

            return langchain_documents
        except Exception as e:
            logger.error(f"Error getting documents by IDs from Moorcheh: {e}")
            raise
