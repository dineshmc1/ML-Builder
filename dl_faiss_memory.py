import faiss
import numpy as np
import os
import json

class ModalityFAISSMemory:
    def __init__(self, modality):
        self.modality = modality
        self.index_path = f"dl_memory_{modality}.faiss"
        self.metadata_path = f"dl_metadata_{modality}.json"
        
        # Load or create FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # 100D embeddings (after PCA)
            self.index = faiss.IndexFlatL2(100)
            self.metadata = []
    
    def add(self, embedding_100d, dataset_name, best_params, accuracy):
        """Add a new result to the modality-specific index"""
        embedding = np.array([embedding_100d], dtype=np.float32)
        self.index.add(embedding)
        
        self.metadata.append({
            "dataset": dataset_name,
            "best_params": best_params,
            "accuracy": accuracy
        })
        
        # Save to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def search(self, query_embedding_100d, top_k=3):
        """Find most similar past datasets"""
        # Return empty if the index is empty
        if self.index.ntotal == 0:
            return []
            
        query = np.array([query_embedding_100d], dtype=np.float32)
        search_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:  # Valid result
                results.append(self.metadata[idx])
        
        return results
