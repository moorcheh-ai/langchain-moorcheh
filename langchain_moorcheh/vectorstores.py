from langchain.vectorstores import VectorStore
from moorcheh_sdk import MoorchehClient

import logging
import os
import uuid
import asyncio
from typing import Any, List, Optional, Literal, Tuple, Type, TypeVar, Sequence
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from uuid import uuid4

logger = logging.getLogger(__name__)

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

        moorcheh_docs_to_upload: List[dict] = []
        assigned_ids: List[str] = []

        for i, doc in enumerate(documents):
            if ids is not None:
                doc_id = str(ids[i])
            else:
                doc_id = (
                    str(getattr(doc, "id"))
                    if getattr(doc, "id", None) is not None
                    else (str(getattr(doc, "id_")) if getattr(doc, "id_", None) is not None else str(uuid4()))
                )

            metadata = (doc.metadata or {}).copy()

            moorcheh_doc = {
                "id": doc_id,
                "text": doc.page_content,
                "metadata": metadata,
            }
            moorcheh_docs_to_upload.append(moorcheh_doc)
            assigned_ids.append(doc_id)

        self._client.upload_documents(
            namespace_name=self.namespace,
            documents=moorcheh_docs_to_upload,
        )
        return assigned_ids

    def upload_vectors(
        self,
        vectors: List[Tuple[str, List[float], Optional[dict]]],   
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
                page_content = result.get("text") or result.get("text_content", "")
                # Extract id from possible keys
                doc_id = result.get("id")
                if doc_id is None:
                    doc_id = result.get("ids")
                    if isinstance(doc_id, list):
                        doc_id = doc_id[0] if doc_id else None
                if doc_id is not None:
                    doc_id = str(doc_id)

                meta_raw = result.get("metadata") or {}
                if isinstance(meta_raw, dict) and "metadata" in meta_raw and isinstance(meta_raw["metadata"], dict) and len(meta_raw) == 1:
                    metadata = meta_raw["metadata"]
                else:
                    metadata = meta_raw

                documents.append(Document(page_content=page_content, metadata=metadata, id=doc_id))

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
              try:
                  score = float(result.get("score", 0.0))
              except Exception:
                  score = 0.0

              page_content = result.get("text") or result.get("text_content", "")

              doc_id = result.get("id")
              if doc_id is None:
                  doc_id = result.get("ids")
                  if isinstance(doc_id, list):
                      doc_id = doc_id[0] if doc_id else None
              if doc_id is not None:
                  doc_id = str(doc_id)

              meta_raw = result.get("metadata") or {}
              if isinstance(meta_raw, dict) and "metadata" in meta_raw and isinstance(meta_raw["metadata"], dict) and len(meta_raw) == 1:
                  metadata = meta_raw["metadata"]
              else:
                  metadata = meta_raw

              scored_langchain_docs.append(
                  (Document(page_content=page_content, metadata=metadata, id=doc_id), score)
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

    
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        if not ids:
            return []
       
        try:
            resp = self._client.get_documents(
                namespace_name=self.namespace,
                ids=ids
            )
            items = resp.get("documents") or resp.get("items") or []
            docs: List[Document] = []
            for d in items:
                text = d.get("text") or d.get("text_content") or ""
                meta_raw = d.get("metadata") or {}
                if isinstance(meta_raw, dict) and "metadata" in meta_raw and isinstance(meta_raw["metadata"], dict) and len(meta_raw) == 1:
                    meta = meta_raw["metadata"]
                else:
                    meta = meta_raw

                doc_id = d.get("id")
                if doc_id is None:
                    doc_id = d.get("ids")
                    if isinstance(doc_id, list):
                        doc_id = doc_id[0] if doc_id else None
                if doc_id is None:
                    alt = meta.get("id") or meta.get("ids")
                    if isinstance(alt, list):
                        alt = alt[0] if alt else None
                        doc_id = alt
                if doc_id is not None:
                     doc_id = str(doc_id)

            docs.append(Document(page_content=text, metadata=meta, id=doc_id))

            by_id = {doc.id: doc for doc in docs if doc.id is not None}
            return [by_id[i] for i in ids if i in by_id]

        except Exception:
            time.sleep(0.1)

        return []
        
    """ Async Methods """

            
    async def adelete(
            self,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> Optional[bool]:
            if not ids:
                return False

            if self.namespace_type == "text":
                print(f"Deleting {len(ids)} documents from Moorcheh (text namespace)...")
                await asyncio.to_thread(
                    self._client.delete_documents,
                    namespace_name=self.namespace,
                    ids=ids,
                )
            elif self.namespace_type == "vector":
                print(f"Deleting {len(ids)} vectors from Moorcheh (vector namespace)...")
                await asyncio.to_thread(
                    self._client.delete_vectors,
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

            instance = cls(embedding=embedding, **kwargs)
            await instance.aadd_documents(documents=documents, ids=ids)
            return instance


    async def aadd_documents(
            self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs: Any
        ) -> List[str]:
            if not documents:
                return []

            if ids is not None and len(ids) != len(documents):
                raise ValueError("Number of IDs must match number of documents if provided.")

            moorcheh_docs_to_upload: List[dict] = []
            assigned_ids: List[str] = []

            for i, doc in enumerate(documents):
                if ids is not None:
                    doc_id = str(ids[i])
                else:
                    doc_id = (
                        str(getattr(doc, "id"))
                        if getattr(doc, "id", None) is not None
                        else (str(getattr(doc, "id_", None)) if getattr(doc, "id_", None) is not None else str(uuid4()))
                    )
                assigned_ids.append(doc_id)

                metadata = (doc.metadata or {}).copy()

                moorcheh_docs_to_upload.append(
                    {"id": doc_id, "text": doc.page_content, "metadata": metadata}
                )

            await asyncio.to_thread(
                self._client.upload_documents,
                namespace_name=self.namespace,
                documents=moorcheh_docs_to_upload,
            )
            return assigned_ids


    async def aupload_vectors(
            self,
            vectors: List[Tuple[str, List[float], Optional[dict]]],   
        ) -> List[str]:
            if self.namespace_type != "vector":
                raise ValueError("upload_vectors is only valid for 'vector' namespaces.")

            payload: List[dict] = []
            ids: List[str] = []

            for doc_id, vector, metadata in vectors:
                payload.append({"id": doc_id, "vector": vector, "metadata": metadata or {}})
                ids.append(doc_id)

            await asyncio.to_thread(
                self._client.upload_vectors,
                namespace_name=self.namespace,
                vectors=payload,
            )
            return ids


    async def asimilarity_search(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
        ) -> List[Document]:
            search_results = await asyncio.to_thread(
                self._client.search,
                namespaces=[self.namespace],
                query=query,
                top_k=k,
                **kwargs,
            )

            raw_items = (
                search_results.get("results")
                or search_results.get("items")
                or search_results.get("documents")
                or search_results.get("matches")
                or []
            )

            documents: List[Document] = []
            for result in raw_items:
                page_content = result.get("text") or result.get("text_content", "")

                doc_id = result.get("id")
                if doc_id is None:
                    alt = result.get("ids")
                    if isinstance(alt, list):
                        doc_id = alt[0] if alt else None
                doc_id = str(doc_id) if doc_id is not None else None

                meta_raw = result.get("metadata") or {}
                if (
                    isinstance(meta_raw, dict)
                    and "metadata" in meta_raw
                    and isinstance(meta_raw["metadata"], dict)
                    and len(meta_raw) == 1
                ):
                    metadata = meta_raw["metadata"]
                else:
                    metadata = meta_raw

                documents.append(Document(page_content=page_content, metadata=metadata, id=doc_id))

            return documents


    async def asimilarity_search_with_score(
            self, query: str, k: int = 4, **kwargs: Any
        ) -> List[Tuple[Document, float]]:
            search_results = await asyncio.to_thread(
                self._client.search,
                namespaces=[self.namespace],
                query=query,
                top_k=k,
                **kwargs,
            )

            raw_items = (
                search_results.get("results")
                or search_results.get("items")
                or search_results.get("documents")
                or search_results.get("matches")
                or []
            )

            out: List[Tuple[Document, float]] = []
            for r in raw_items:
                try:
                    score = float(r.get("score", 0.0))
                except Exception:
                    score = 0.0

                text = r.get("text") or r.get("text_content", "")

                doc_id = r.get("id")
                if doc_id is None:
                    alt = r.get("ids")
                    if isinstance(alt, list):
                        doc_id = alt[0] if alt else None
                doc_id = str(doc_id) if doc_id is not None else None

                meta_raw = r.get("metadata") or {}
                if (
                    isinstance(meta_raw, dict)
                    and "metadata" in meta_raw
                    and isinstance(meta_raw["metadata"], dict)
                    and len(meta_raw) == 1
                ):
                    metadata = meta_raw["metadata"]
                else:
                    metadata = meta_raw

                out.append((Document(page_content=text, metadata=metadata, id=doc_id), score))

            return out


    async def agenerative_answer(self, query: str, k: int = 4, **kwargs: Any):
            result = await asyncio.to_thread(
                self._client.get_generative_answer,
                namespace=self.namespace,
                query=query,
                top_k=k,
                ai_model="anthropic.claude-3-7-sonnet-20250219-v1:0",
                **kwargs,
            )
            return result.get("answer", "")


    async def aget_by_ids(self, ids: List[str]) -> List[Document]:
            if not ids:
                return []
                
            try:
                response = await asyncio.to_thread(
                    self._client.get_documents,
                    namespace_name=self.namespace,
                    ids=ids,
                )
                
                items = response.get("documents") or response.get("items") or []
                lc_docs: List[Document] = []

                for d in items:
                    text = d.get("text") or d.get("text_content") or ""

                    meta_raw = d.get("metadata") or {}
                    if (
                        isinstance(meta_raw, dict)
                        and "metadata" in meta_raw
                        and isinstance(meta_raw["metadata"], dict)
                        and len(meta_raw) == 1
                    ):
                        meta = meta_raw["metadata"]
                    else:
                        meta = meta_raw

                    doc_id = d.get("id")
                    if doc_id is None:
                        alt = d.get("ids")
                        if isinstance(alt, list):
                            doc_id = alt[0] if alt else None
                    if doc_id is None:
                        alt = meta.get("id") or meta.get("ids")
                        if isinstance(alt, list):
                            alt = alt[0] if alt else None
                        doc_id = alt
                    doc_id = str(doc_id) if doc_id is not None else None

                    lc_docs.append(Document(page_content=text, metadata=meta, id=doc_id))

                by_id = {doc.id: doc for doc in lc_docs if doc.id is not None}
                return [by_id[i] for i in ids if i in by_id]

                except Exception:
                    await asyncio.sleep(0.1)

            return []
