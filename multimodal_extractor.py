import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import warnings
from dotenv import load_dotenv

load_dotenv() # Load .env file if it exists

# Optional: If the user provides a token, use it. If not, ignore the warning and use anonymous.
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
else:
    # Suppress the annoying warning for anonymous users
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

# Suppress annoying warnings
warnings.filterwarnings('ignore')

class UniversalEmbedder:
    def __init__(self, device='cpu', batch_size=32, domain='general'):
        self.device = device
        self.batch_size = batch_size
        self.domain = domain
        
        # Lazy loading of models to save memory
        self.vision_model = None
        self.vision_processor = None
        self.text_model = None
        
    def embed_directory(self, data_path, modality):
        """
        Scans a directory (assuming subfolders are class labels),
        extracts embeddings, applies PCA, and returns X (DataFrame) and y (Series).
        """
        # --- NEW: AUTO-DETECT TRAIN/TEST SPLITS ---
        root_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        # If the folder contains standard ML split names, automatically dive into 'train'
        if set(['train', 'test', 'val']).intersection(set([d.lower() for d in root_dirs])):
            train_path = os.path.join(data_path, 'train')
            if os.path.exists(train_path):
                print(f"[Embedder] Detected Train/Test split structure. Automatically routing to: {train_path}")
                data_path = train_path
        # --------------------------------------------

        print(f"[UniversalEmbedder] Starting extraction for modality: {modality.upper()}")
        
        # Find all files and their class labels (subfolder names)
        files = []
        labels = []
        for root, _, filenames in os.walk(data_path):
            label = os.path.basename(root)
            if root == data_path:
                label = 'unknown' # Files in root dir
                
            for f in filenames:
                # Basic filter for valid extensions
                ext = os.path.splitext(f)[1].lower()
                if modality == 'vision' and ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']: continue
                if modality == 'text' and ext not in ['.txt', '.md']: continue
                if modality == 'audio' and ext not in ['.wav', '.mp3', '.flac', '.ogg']: continue
                if modality == 'video' and ext not in ['.mp4', '.avi', '.mov', '.mkv']: continue
                
                files.append(os.path.join(root, f))
                labels.append(label)
                
        if not files:
            raise ValueError(f"No valid {modality} files found in {data_path}")
            
        print(f"[UniversalEmbedder] Found {len(files)} files across {len(set(labels))} classes.")
        
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(files), self.batch_size), desc=f"Extracting {modality} embeddings"):
            batch_files = files[i:i+self.batch_size]
            
            if modality == 'vision':
                batch_emb = self._process_vision_batch(batch_files)
            elif modality == 'text':
                batch_emb = self._process_text_batch(batch_files)
            elif modality == 'audio':
                batch_emb = self._process_audio_batch(batch_files)
            elif modality == 'video':
                batch_emb = self._process_video_batch(batch_files)
            else:
                raise ValueError(f"Unsupported modality: {modality}")
                
            embeddings.append(batch_emb)
            
        # Combine all batches
        X_raw = np.vstack(embeddings)
        y = pd.Series(labels)
        
        print(f"[UniversalEmbedder] Raw embeddings shape: {X_raw.shape}")
        
        # PCA Dimensionality Reduction
        # Audio MFCCs are already small (~40), so we only reduce large embeddings
        if X_raw.shape[1] > 100:
            n_components = min(100, X_raw.shape[0]) # Can't have more components than samples
            print(f"[UniversalEmbedder] Applying PCA to reduce dimensions from {X_raw.shape[1]} to {n_components}...")
            pca = PCA(n_components=n_components, random_state=42)
            X_reduced = pca.fit_transform(X_raw)
            print(f"[UniversalEmbedder] PCA complete. Explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")
        else:
            X_reduced = X_raw
            
        # Convert to Pandas DataFrame
        feature_names = [f"feat_{i}" for i in range(X_reduced.shape[1])]
        X_df = pd.DataFrame(X_reduced, columns=feature_names)
        
        return X_df, y

    def _process_vision_batch(self, file_paths):
        from PIL import Image
        import torch
        
        if self.vision_model is None:
            from transformers import AutoProcessor, AutoModel
            from domain_registry import get_vision_model_config
            
            cfg = get_vision_model_config(self.domain, "clip")
            model_id = cfg["model_id"]
            
            print(f"\\n[UniversalEmbedder] Loading domain-specific vision model ({self.domain}): {model_id}...")
            
            # Using AutoProcessor / AutoModel to handle diverse architectures (CLIP, BEiT, TrOCR)
            self.vision_processor = AutoProcessor.from_pretrained(model_id)
            self.vision_model = AutoModel.from_pretrained(model_id).to(self.device)
            self.vision_model.eval()
            
        images = []
        for path in file_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Fallback to a blank image
                images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                
        # TrOCR processor expects 'images', while others might expect 'images' or 'pixel_values'
        # AutoProcessor handles this mostly, but we use 'images' explicitly
        inputs = self.vision_processor(images=images, return_tensors="pt")
        # Padding might be required depending on the exact processor
        if hasattr(self.vision_processor, 'pad'):
             inputs = self.vision_processor.pad(inputs, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            # Use get_image_features to only run the vision tower (avoids input_ids error)
            outputs = self.vision_model.get_image_features(**inputs) 
            
            # FIX: Extract the actual tensor from the output object
            if isinstance(outputs, torch.Tensor):
                image_features = outputs
            elif hasattr(outputs, 'image_embeds'):
                # Specific to CLIP Vision Model
                image_features = outputs.image_embeds 
            elif hasattr(outputs, 'last_hidden_state'):
                # Specific to standard HF Vision Transformers (ViT)
                image_features = outputs.last_hidden_state[:, 0, :] 
            else:
                # Fallback for other models
                image_features = outputs[0] 
            
        return image_features.cpu().numpy()
        
    def _process_text_batch(self, file_paths):
        if self.text_model is None:
            from sentence_transformers import SentenceTransformer
            print("\n[UniversalEmbedder] Loading SentenceTransformer...")
            self.text_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            
        texts = []
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read()[:5000]) # Cap length for speed
            except Exception as e:
                texts.append("")
                
        embeddings = self.text_model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return embeddings

    def _process_audio_batch(self, file_paths):
        import librosa
        batch_emb = []
        
        for path in file_paths:
            try:
                # Load audio, resample to 22050Hz, max 30 seconds to avoid memory spikes
                y, sr = librosa.load(path, sr=22050, duration=30.0)
                
                if len(y) == 0:
                    batch_emb.append(np.zeros(40))
                    continue
                    
                # Extract 40 MFCCs
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                
                # Average across the time axis to get a 1D vector per file
                mfccs_mean = np.mean(mfccs.T, axis=0)
                batch_emb.append(mfccs_mean)
                
            except Exception as e:
                print(f"Error processing audio {path}: {e}")
                batch_emb.append(np.zeros(40))
                
        return np.vstack(batch_emb)

    def _process_video_batch(self, file_paths):
        import cv2
        from PIL import Image
        import torch
        
        if self.vision_model is None:
            from transformers import CLIPProcessor, CLIPModel
            print("\n[UniversalEmbedder] Loading CLIP model for Video Frames...")
            self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.vision_model.eval()
            
        batch_emb = []
        
        for path in file_paths:
            frames = []
            try:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 24 # Fallback
                
                frame_count = 0
                success, frame = cap.read()
                
                while success:
                    # Extract 1 frame per second
                    if frame_count % int(fps) == 0:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame_rgb))
                        
                        # Cap at 10 frames max per video to save time/memory
                        if len(frames) >= 10:
                            break
                            
                    success, frame = cap.read()
                    frame_count += 1
                cap.release()
                
            except Exception as e:
                print(f"Error processing video {path}: {e}")
                
            if not frames:
                # Blank fallback
                batch_emb.append(np.zeros(512))
                continue
                
            # Embed all extracted frames
            inputs = self.vision_processor(images=frames, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.vision_model.get_image_features(**inputs)
                if isinstance(outputs, torch.Tensor):
                    frame_features = outputs
                elif hasattr(outputs, 'image_embeds'):
                    frame_features = outputs.image_embeds
                elif hasattr(outputs, 'last_hidden_state'):
                    frame_features = outputs.last_hidden_state[:, 0, :]
                else:
                    frame_features = outputs[0]
                
            # Average frame embeddings for the video representation
            video_embedding = torch.mean(frame_features, dim=0)
            batch_emb.append(video_embedding.cpu().numpy())
            
        return np.vstack(batch_emb)
