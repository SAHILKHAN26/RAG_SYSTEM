# Migration from ChromaDB to FAISS

## Why FAISS?

FAISS (Facebook AI Similarity Search) was chosen to replace ChromaDB due to:

✅ **No hnswlib dependency issues** - FAISS has cleaner dependencies  
✅ **Better performance** - Faster similarity search  
✅ **Production-ready** - Used by Facebook/Meta in production  
✅ **More stable** - Fewer installation problems  
✅ **Flexible** - Supports multiple index types  

## Changes Made

### 1. Vector Store Implementation
- **File**: `src/core/services/vector_store.py`
- **Changed**: Complete rewrite to use FAISS instead of ChromaDB
- **Key Differences**:
  - Uses `faiss.IndexFlatL2` for exact L2 distance search
  - Metadata stored separately in pickle file
  - Manual index persistence (save/load)
  - Deletion requires index rebuild

### 2. Configuration
- **File**: `src/core/constant.py`
- **Changed**: `CHROMA_PERSIST_DIR` → `VECTOR_STORE_DIR`
- **Default**: `./faiss_index` instead of `./chroma_db`

### 3. Dependencies
- **File**: `requirements.txt`
- **Removed**: `chromadb==0.4.22`
- **Added**: `faiss-cpu==1.7.4`

### 4. Environment Variables
- **File**: `.env.example`
- **Changed**: `CHROMA_PERSIST_DIR` → `VECTOR_STORE_DIR`

## Installation

```bash
# Install FAISS
pip install faiss-cpu==1.7.4

# Or install all dependencies
pip install -r requirements.txt
```

## Usage

The API remains **exactly the same**! No code changes needed in your application.

```python
from src.core.services.vector_store import get_vector_store

# Get vector store instance (now using FAISS)
vector_store = get_vector_store()

# Add documents (same as before)
chunks = vector_store.chunk_text(text, document_id)
vector_store.add_document_chunks(chunks)

# Search (same as before)
results = vector_store.search(query, top_k=3)

# Delete (same as before)
vector_store.delete_document(document_id)
```

## FAISS Index Types

The current implementation uses `IndexFlatL2` for exact search. For larger datasets, you can modify to use:

### IndexIVFFlat (Faster, Approximate)
```python
# In _create_index method
quantizer = faiss.IndexFlatL2(self.dimension)
self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
self.index.train(training_vectors)  # Requires training
```

### IndexHNSWFlat (Fast, High Recall)
```python
# In _create_index method
self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors
```

## Performance Comparison

| Feature | ChromaDB | FAISS |
|---------|----------|-------|
| Installation | ❌ hnswlib issues | ✅ Clean |
| Search Speed | Fast | **Faster** |
| Memory Usage | Moderate | **Lower** |
| Scalability | Good | **Excellent** |
| Production Use | Good | **Battle-tested** |

## File Structure

```
faiss_index/
├── bot_gpt_documents.index      # FAISS index file
└── bot_gpt_documents_metadata.pkl  # Metadata pickle file
```

## Migration Steps (If You Have Existing ChromaDB Data)

If you already have data in ChromaDB and want to migrate:

```python
import chromadb
from src.core.services.vector_store import get_vector_store

# 1. Load old ChromaDB data
old_client = chromadb.PersistentClient(path="./chroma_db")
old_collection = old_client.get_collection("bot_gpt_documents")

# 2. Get all data
all_data = old_collection.get()

# 3. Add to new FAISS store
vector_store = get_vector_store()
vector_store.add_documents(
    texts=all_data['documents'],
    metadatas=all_data['metadatas'],
    ids=all_data['ids']
)

print(f"Migrated {len(all_data['ids'])} chunks to FAISS")
```

## Troubleshooting

### Issue: "No module named 'faiss'"
```bash
pip install faiss-cpu==1.7.4
```

### Issue: "Cannot import name 'IndexFlatL2'"
```bash
# Uninstall any conflicting packages
pip uninstall faiss faiss-gpu -y

# Reinstall faiss-cpu
pip install faiss-cpu==1.7.4
```

### Issue: Old ChromaDB directory still exists
```bash
# Safe to delete after migration
rm -rf ./chroma_db
```

## Benefits You'll See

1. **No more hnswlib errors** during installation
2. **Faster search** for large document collections
3. **Lower memory usage** with optimized indices
4. **Better production stability**
5. **Easier deployment** (fewer dependencies)

## Next Steps

1. Update your `.env` file:
   ```bash
   VECTOR_STORE_DIR=./faiss_index
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

Everything else works exactly the same! 🎉
