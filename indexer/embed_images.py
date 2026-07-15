"""
Generates FashionCLIP embeddings for a set of images.
Uses patrickjohncyh/fashion-clip (loaded directly via transformers,
since the fashion-clip pip package has broken auth-token compatibility
with current transformers versions).
"""
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_fashion_clip(device=None):
    """Loads FashionCLIP model + processor onto the given device (auto-detects GPU)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    model.eval()
    return model, processor, device


def extract_tensor(output):
    """Handles both plain-tensor and wrapped-object return types across transformers versions."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, 'image_embeds'):
        return output.image_embeds
    if hasattr(output, 'pooler_output'):
        return output.pooler_output
    raise TypeError(f"Unexpected output type: {type(output)}")


def embed_images(image_metadata, images_dir, model, processor, device, batch_size=32):
    """
    Given {image_id: {file_name, ...}}, returns:
      - embeddings: np.ndarray of shape (N, 512), L2-normalized
      - valid_ids: list of image_ids in the same order as embeddings
    """
    image_ids_list = list(image_metadata.keys())
    all_embeddings = []
    valid_ids = []

    for i in range(0, len(image_ids_list), batch_size):
        batch_ids = image_ids_list[i:i + batch_size]
        batch_images, batch_valid_ids = [], []

        for img_id in batch_ids:
            file_name = image_metadata[img_id]['file_name']
            try:
                img = Image.open(f"{images_dir}/{file_name}").convert('RGB')
                batch_images.append(img)
                batch_valid_ids.append(img_id)
            except Exception as e:
                print(f"Skipping {img_id}: {e}")

        if not batch_images:
            continue

        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            raw_output = model.get_image_features(**inputs)
            image_features = extract_tensor(raw_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        all_embeddings.append(image_features.cpu().numpy())
        valid_ids.extend(batch_valid_ids)

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings.astype('float32'), valid_ids


def embed_text_query(query_text, model, processor, device):
    """Converts a natural language query into the same embedding space as the images."""
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        raw_output = model.get_text_features(**inputs)
        text_features = extract_tensor(raw_output) if not isinstance(raw_output, torch.Tensor) else raw_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype('float32')
