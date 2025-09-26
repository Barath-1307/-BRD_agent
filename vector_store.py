# ----------------------------------------------------------------------
# File: build_vector_store.py
#
# Purpose:
#   Builds a FAISS-based vector database from previously crawled JSON
#   documentation files. Converts extracted page text into embeddings
#   using a SentenceTransformer model and saves both the index and the
#   raw text for later semantic search / retrieval.
#
# Key Features:
#   - Recursively extracts text fields from arbitrary JSON structures.
#   - Uses the "all-MiniLM-L6-v2" model from SentenceTransformers to
#     generate dense embeddings for text chunks.
#   - Normalizes embeddings for cosine similarity search.
#   - Builds a FAISS L2 index of all text embeddings.
#   - Saves the FAISS index (`chunks.index`) and original texts
#     (`texts.pkl`) into a `vector_store/` folder for reuse.
#
# Usage:
#   python build_vector_store.py
#
#   - Expects JSON files in the folder: ./crawled_json_files
#   - Creates a FAISS index and saves it to: ./vector_store/chunks.index
#   - Stores all original text chunks to: ./vector_store/texts.pkl
#
# Output:
#   - Console logs embedding shape, number of vectors, and save paths.
# ----------------------------------------------------------------------


import os
import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def extract_texts(data):
    if isinstance(data, dict) and "text_content" in data and isinstance(data["text_content"], str):
        return [data["text_content"]]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if "text" in data[0]:
            return [item["text"] for item in data if "text" in item]
        elif "content" in data[0]:
            return [item["content"] for item in data if "content" in item]
        elif "body" in data[0]:
            return [item["body"] for item in data if "body" in item]
    if isinstance(data, dict):
        for key in ["text", "content", "body"]:
            if key in data and isinstance(data[key], str):
                return [data[key]]
            if key in data and isinstance(data[key], list):
                return data[key]
    if isinstance(data, list) and isinstance(data[0], str):
        return data
    if isinstance(data, dict):
        texts = []
        for v in data.values():
            if isinstance(v, str):
                texts.append(v)
            elif isinstance(v, (list, dict)):
                texts.extend(extract_texts(v))
        return texts
    if isinstance(data, list):
        texts = []
        for v in data:
            texts.extend(extract_texts(v))
        return texts
    return []

# 1) Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2) Collect all texts from JSON files
all_texts = []
json_folder = os.path.join(os.getcwd(), 'crawled_json_files')
for filename in os.listdir(json_folder):
    with open(os.path.join(json_folder, filename), encoding='utf-8') as file:
        data = json.load(file)
        texts = extract_texts(data)
        if texts and isinstance(texts, list) and isinstance(texts[0], str):
            all_texts.extend(texts)
        else:
            print(f"Skipping {filename}: Unrecognized JSON structure.")

if not all_texts:
    print("No texts found to index.")
    exit()

# 3) Generate embeddings
embeddings = model.encode(all_texts, convert_to_numpy=True)
print("Embedding shape:", embeddings.shape)

# 4) Normalize embeddings (for cosine similarity search)
faiss.normalize_L2(embeddings)

# 5) Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print("Number of vectors in index:", index.ntotal)

# 6) Create a folder to store vector DB
output_folder = os.path.join(os.getcwd(), "vector_store")
os.makedirs(output_folder, exist_ok=True)

# 7) Save index
index_path = os.path.join(output_folder, "chunks.index")
faiss.write_index(index, index_path)
print(f"Saved FAISS index to {index_path}")

# 8) Save texts for lookup
texts_path = os.path.join(output_folder, "texts.pkl")
with open(texts_path, "wb") as f:
    pickle.dump(all_texts, f)
print(f"Saved all texts to {texts_path}")
