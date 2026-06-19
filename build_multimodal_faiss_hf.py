import os
import shutil
import numpy as np
from datasets import load_dataset
from multimodal_extractor import UniversalEmbedder
from dl_faiss_memory import ModalityFAISSMemory
from sklearn.decomposition import PCA

# --- CONFIGURATION: Curated Hugging Face Dataset IDs ---
# We only need 5-10 high-quality datasets per modality to prove the concept.
HF_DATASETS = {
    "vision": [
        "cifar10",               # 10 classes, 60k images
        "fashion_mnist",         # 10 classes, 70k images
        "stanford_dogs",         # 120 dog breeds
        "oxford_iiit_pet",       # 37 pet breeds
        "beans",                 # 3 classes (plant disease)
        "pothole",               # 2 classes (road damage)
    ],
    "audio": [
        "gtzan_genre",           # 10 music genres
        "common_voice",          # Speech (we will use a small subset)
        "freesound_one_shot_audio", # Various sounds
    ],
    "text": [
        "imdb",                  # Movie reviews (sentiment)
        "ag_news",               # News classification (4 classes)
        "amazon_polarity",       # Product reviews
        "yahoo_answers_topics",  # Q&A topics
    ]
}

TEMP_DIR = "temp_hf_datasets"
os.makedirs(TEMP_DIR, exist_ok=True)

def build_memory():
    print("="*50)
    print("AUTOMATED MULTI-MODAL FAISS BUILDER (HuggingFace)")
    print("="*50)
    
    embedder = UniversalEmbedder(device='cpu') # Change to 'cuda' if you have a GPU!
    
    for modality, dataset_ids in HF_DATASETS.items():
        print(f"\n Processing Modality: {modality.upper()}")
        memory = ModalityFAISSMemory(modality)
        
        for ds_id in dataset_ids:
            print(f"\n📂 Downloading & Processing: {ds_id}")
            dataset_path = os.path.join(TEMP_DIR, ds_id.replace("/", "_"))
            
            try:
                # 1. Load dataset from Hugging Face
                # We only load the 'train' split to save time
                dataset = load_dataset(ds_id, split="train", trust_remote_code=True)
                
                # 2. Save to temporary folder structure (so your embedder can read it)
                if os.path.exists(dataset_path):
                    shutil.rmtree(dataset_path)
                
                # Note: Saving HF datasets to folders requires specific handling per dataset.
                # For simplicity in this script, we will process the HF dataset object directly 
                # if your embedder supports it, OR we save images to folders.
                
                # --- SIMPLIFIED APPROACH FOR VISION ---
                if modality == "vision" and "image" in dataset.features:
                    os.makedirs(dataset_path, exist_ok=True)
                    labels = dataset.features["label"].names
                    for label in labels:
                        os.makedirs(os.path.join(dataset_path, label), exist_ok=True)
                        
                    # Save first 100 images per class to keep it fast
                    counts = {label: 0 for label in labels}
                    for item in dataset:
                        label_name = labels[item["label"]]
                        if counts[label_name] < 100:
                            img_path = os.path.join(dataset_path, label_name, f"{counts[label_name]}.jpg")
                            item["image"].save(img_path)
                            counts[label_name] += 1
                            
                    # 3. Extract Embeddings
                    X, y = embedder.embed_directory(dataset_path, modality)
                    
                    # 4. PCA & Save
                    pca = PCA(n_components=100, random_state=42)
                    X_pca = pca.fit_transform(X)
                    rep_embedding = np.mean(X_pca, axis=0)
                    
                    memory.add(
                        embedding_100d=rep_embedding,
                        dataset_name=ds_id,
                        best_params={'dl_num_layers': 2, 'dl_hidden_dim': 64}, # Default warm-start
                        accuracy=0.85 # Dummy accuracy for build phase
                    )
                    print(f"✅ Successfully added {ds_id} to Vision FAISS.")
                    
                # TODO: Add similar logic for 'audio' and 'text' modalities here
                # (Audio requires saving .wav files, Text requires saving .txt files)
                
            except Exception as e:
                print(f"❌ Failed to process {ds_id}: {e}")

if __name__ == "__main__":
    build_memory()
    # Cleanup temp files
    # shutil.rmtree(TEMP_DIR) 
