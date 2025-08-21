# 🦜️🔗 LangChain Moorcheh

This repository contains the LangChain integration with Moorcheh, a powerful vector database for AI applications.

## Installation

```bash
pip install -U langchain-moorcheh
```

## Configuration

Set your Moorcheh API key as an environment variable:

```bash
export MOORCHEH_API_KEY="your-api-key"
```

## Vector Stores

The `MoorchehVectorStore` class allows you to use Moorcheh VectorDB alongside LangChain:

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

## Features

- **Vector Storage**: Store and retrieve embeddings with high performance
- **Namespace Management**: Organize your data with flexible namespace structures
- **LangChain Integration**: Seamlessly integrate with the LangChain ecosystem
- **Async Support**: Full asynchronous operation support
- **Metadata Filtering**: Advanced querying with metadata support

## Documentation

For more detailed information, visit:
- [PyPI Package](https://pypi.org/project/langchain-moorcheh/)
- [Source Code](https://github.com/langchain-ai/langchain-moorcheh)
- [LangChain Documentation](https://python.langchain.com/)

## Contributing

We welcome contributions! Please see our [contributing guidelines](https://github.com/langchain-ai/langchain-moorcheh/blob/main/CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
