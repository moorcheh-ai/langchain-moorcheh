# langchain-moorcheh

This package contains the LangChain integration with Moorcheh

## Installation

```bash
pip install -U langchain-moorcheh
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatMoorcheh` class exposes chat models from Moorcheh.

```python
from langchain_moorcheh import ChatMoorcheh

llm = ChatMoorcheh()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`MoorchehEmbeddings` class exposes embeddings from Moorcheh.

```python
from langchain_moorcheh import MoorchehEmbeddings

embeddings = MoorchehEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`MoorchehLLM` class exposes LLMs from Moorcheh.

```python
from langchain_moorcheh import MoorchehLLM

llm = MoorchehLLM()
llm.invoke("The meaning of life is")
```
