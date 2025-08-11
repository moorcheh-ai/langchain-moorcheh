# Langchain-Moorcheh

This package contains the LangChain integration with Moorcheh.

## Installation

```bash
pip install -U langchain-moorcheh
```

And you should configure credentials by setting the following environment variables:

```bash
export MOORCHEH_API_KEY="your-api-key"
```


## Vector Stores

`Moorcheh` vector store class allows you to use Moorcheh VectorDB alongside langchain.

```python
from langchain_moorcheh import MoorchehVectorStore

vector_store = MoorchehVectorStore.from_texts(
    texts=texts,
    embedding=embedding_model,
    api_key=MOORCHEH_API_KEY,
    namespace=NAMESPACE_NAME,
    namespace_type=NAMESPACE_TYPE,
)

```
