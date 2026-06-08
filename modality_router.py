import os

class ModalityRouter:
    def __init__(self, data_path):
        self.data_path = data_path
        self.modality = self._detect_modality()

    def _detect_modality(self):
        # We need to search recursively in case the user provided a folder containing class subfolders
        exts = []
        for root, _, files in os.walk(self.data_path):
            for f in files:
                exts.append(os.path.splitext(f)[1].lower())
        
        if not exts:
            return "unknown"
            
        most_common_ext = max(set(exts), key=exts.count)
        
        if most_common_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            return "vision"
        elif most_common_ext in ['.txt', '.csv', '.json', '.md']:
            return "text"
        elif most_common_ext in ['.wav', '.mp3', '.flac', '.ogg']:
            return "audio"
        elif most_common_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return "video"
        else:
            return "tabular" # Fallback to CSV/Tabular

    def get_modality(self):
        return self.modality
