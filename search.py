# ----------------------------------------------------------------------
# File: query_vector_store.py
#
# Purpose:
#   Provides a wrapper class (`VectorDB`) to query a FAISS vector index
#   built from crawled documentation. Lets you run semantic search
#   against previously embedded text chunks.
#
# Key Features:
#   - Loads a FAISS index (`chunks.index`) and corresponding text chunks
#     (`texts.pkl`) from disk.
#   - Uses SentenceTransformers ("all-MiniLM-L6-v2") to embed new queries.
#   - Normalizes embeddings for cosine similarity.
#   - Returns the most relevant text chunks with similarity scores.
#   - Includes a runnable `__main__` section for testing queries and
#     saving results to `vector_store/search_results.txt`.
#
# Usage:
#   python query_vector_store.py
#
#   - Make sure `vector_store/chunks.index` and `vector_store/texts.pkl`
#     exist (produced by build_vector_store.py).
#   - Edit the `query` string in `__main__` to change what you’re looking for.
#   - Results are printed to console and written to a results text file.
#
# Example:
#   $ python query_vector_store.py
#   Top 10 results for query: 'Create geofences around locations'
#   1. Score: 0.87
#      <matching text chunk>
#   ...
#
# Output:
#   - Printed results in terminal.
#   - Results file: ./vector_store/search_results.txt
# ----------------------------------------------------------------------


import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, index_path: str, text_path: str, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the FAISS-based vector database.
        Args:
            index_path: Path to saved FAISS index
            text_path: Path to saved text chunks (list of strings)
            model_name: SentenceTransformer model for embeddings
        """
        self.index_path = index_path
        self.text_path = text_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = None
        self.load_index_and_texts()

    def load_index_and_texts(self):
        """Load FAISS index and text chunks from disk"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")
        if not os.path.exists(self.text_path):
            raise FileNotFoundError(f"Text chunks file not found: {self.text_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        # Load text chunks
        with open(self.text_path, "rb") as f:
            self.texts = pickle.load(f)
        
        print(f"Loaded FAISS index with {self.index.ntotal} vectors.")
        print(f"Loaded {len(self.texts)} text chunks.")

    def similarity_search(self, query: str, top_k: int = 3):
        """
        Search for the most similar text chunks to the query.
        Args:
            query: Text query
            top_k: Number of results to return
        Returns:
            List of dicts: [{'text': ..., 'score': ...}, ...]
        """
        if self.index is None or self.texts is None:
            raise RuntimeError("Index or texts not loaded.")
        # cap top_k
        top_k = max(1, min(top_k, max(1, self.index.ntotal)))
        # Encode and normalize query
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        
        # Search FAISS
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue  # handle empty results
            similarity = 1 - dist / 2  # cosine similarity for normalized vectors + IndexFlatL2
            results.append({
                "text": self.texts[idx],
                "score": similarity
            })
        return results

# -------------------
# Usage Example
# -------------------
if __name__ == "__main__":
    # Paths to your saved files (use vector_store folder)
    faiss_index_path = os.path.join(os.getcwd(), "vector_store", "chunks.index")
    text_chunks_path = os.path.join(os.getcwd(), "vector_store", "texts.pkl")
    
    # Load the vector database
    db = VectorDB(index_path=faiss_index_path, text_path=text_chunks_path)
    
    # Perform similarity search
    query = "Create geofences around locations"
    results = db.similarity_search(query, top_k=10)
    
    print(f"Top {len(results)} results for query: '{query}'\n")
    for i, res in enumerate(results, start=1):
        print(f"{i}. Score: {res['score']:.4f}\n{res['text']}\n{'-'*50}")

    # Save results to a text file
    output_path = os.path.join(os.getcwd(), "vector_store", "search_results.txt")
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(f"Top {len(results)} results for query: '{query}'\n\n")
        for i, res in enumerate(results, start=1):
            out_f.write(f"{i}. Score: {res['score']:.4f}\n")
            out_f.write(res['text'].strip() + "\n")
            out_f.write("-" * 50 + "\n")
    print(f"Saved results to {output_path}")