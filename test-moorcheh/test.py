import os
from uuid import uuid4
from langchain_moorcheh import MoorchehVectorStore
from moorcheh_sdk import MoorchehClient
from langchain_core.documents import Document

# ============================
# 1. SETUP (Run once or set in your shell)
# ============================
# Make sure you have your Moorcheh API key set as an environment variable:
# export MOORCHEH_API_KEY="your_actual_key_here"
MOORCHEH_API_KEY = "cP2W58dED991SIKFga7Nj8FHbUKnGI6Fa1gbrY8E"
if not MOORCHEH_API_KEY:
    raise ValueError("❌ MOORCHEH_API_KEY environment variable is not set!")

# Choose a unique namespace name (change this if you want a different one)
NAMESPACE_NAME = "langchain-demo-namespace"
NAMESPACE_TYPE = "text"  # "text" or "vector"

print("🚀 Starting Moorcheh + LangChain full demo...")

# ============================
# 2. CREATE NAMESPACE (programmatically - only needed the first time)
# ============================
print(f"📦 Creating/checking namespace: {NAMESPACE_NAME}")
try:
    with MoorchehClient(api_key=MOORCHEH_API_KEY) as client:
        client.namespaces.create(
            namespace_name=NAMESPACE_NAME,
            type=NAMESPACE_TYPE
        )
    print(f"✅ Namespace '{NAMESPACE_NAME}' is ready!")
except Exception as e:
    # Namespace might already exist - that's fine, continue
    print(f"⚠️  Namespace creation skipped (it probably already exists): {e}")

# ============================
# 3. INITIALIZE LANGCHAIN VECTOR STORE
# ============================
store = MoorchehVectorStore(
    api_key=MOORCHEH_API_KEY,
    namespace=NAMESPACE_NAME,
    namespace_type=NAMESPACE_TYPE
)
print("✅ MoorchehVectorStore initialized!")

# ============================
# 4. ADD SAMPLE DOCUMENTS
# ============================
documents = [
    Document(
        page_content="Brewed a fresh cup of Ethiopian coffee and paired it with a warm croissant.",
        metadata={"source": "blog"},
    ),
    Document(
        page_content="Tomorrow's weather will be sunny with light winds, reaching a high of 78°F.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="Experimenting with LangChain for an AI-powered note-taking assistant!",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="Local bakery donates 500 loaves of bread to the community food bank.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="That concert last night was absolutely unforgettable—what a performance!",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="Check out our latest article: 5 ways to boost productivity while working from home.",
        metadata={"source": "website"},
    ),
    Document(
        page_content="The ultimate guide to mastering homemade pizza dough.",
        metadata={"source": "website"},
    ),
    Document(
        page_content="LangGraph just made multi-agent workflows way easier—seriously impressive!",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="Oil prices rose 3% today after unexpected supply cuts from major exporters.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="I really hope this post doesn't vanish into the digital void…",
        metadata={"source": "tweet"},
    ),
]

# Generate unique IDs
uuids = [str(uuid4()) for _ in range(len(documents))]

print(f"📤 Adding {len(documents)} documents...")
try:
    store.add_documents(documents=documents, ids=uuids)
    print("✅ All documents added successfully!")
except Exception as e:
    print(f"❌ Error adding documents: {e}")

# ============================
# 5. SIMILARITY SEARCH (k=3)
# ============================
print("\n🔍 Running similarity search for 'coffee and breakfast'...")
results = store.similarity_search("coffee and breakfast", k=3)
for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"   Content: {doc.page_content}")
    print(f"   Metadata: {doc.metadata}")

# ============================
# 6. SIMILARITY SEARCH WITH SCORE
# ============================
print("\n📊 Running similarity search with scores for 'weather forecast'...")
results_with_score = store.similarity_search_with_score("weather forecast", k=3)
for doc, score in results_with_score:
    print(f"\nScore: {score:.4f}")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")

# ============================
# 7. GENERATIVE ANSWER (RAG-style)
# ============================
print("\n🧠 Generating AI answer...")
query = "Give me a brief summary of the provided documents"
answer = store.generative_answer(
    query=query,
    ai_model="anthropic.claude-sonnet-4-5-20250929-v1:0"   # You can change to any supported model
)
print(f"\n📝 Question: {query}")
print(f"Answer:\n{answer}")

# ============================
# OPTIONAL: DELETE A SPECIFIC DOCUMENT (example)
# ============================
# print("\n🗑️  Deleting first document as example...")
# store.delete(ids=[uuids[0]])

print("\n🎉 Full demo completed! You now have a working Moorcheh + LangChain vector store.")