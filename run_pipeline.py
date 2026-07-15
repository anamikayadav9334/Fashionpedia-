"""
End-to-end pipeline: builds the index from raw Fashionpedia data, then
runs the 5 required evaluation queries.

Usage (run from inside the fashion-retrieval/ directory):
    python run_pipeline.py

Requires the Fashionpedia dataset downloaded locally - update the paths
below to match your environment (this was developed on Kaggle with the
dataset at /kaggle/input/datasets/deepakdhull/fashionpedia/).
"""
import random
import json
from collections import defaultdict

from indexer.build_metadata import build_image_metadata
from indexer.embed_images import load_fashion_clip, embed_images
from indexer.build_index import build_faiss_index, save_index
from retriever.evaluate import run_evaluation

# ---- Config: update these paths for your environment ----
ANNOTATION_JSON = '/kaggle/input/datasets/deepakdhull/fashionpedia/instances_attributes_train2020.json'
IMAGES_DIR = '/kaggle/input/datasets/deepakdhull/fashionpedia/train2020/train'
TARGET_SIZE = 800
MAX_PER_CATEGORY = 40


def select_balanced_subset(annotation_json_path, target_size=800, max_per_category=40, seed=42):
    """Selects a balanced image subset, capping overrepresented categories
    so rarer garment types still get fair representation."""
    with open(annotation_json_path, 'r') as f:
        data = json.load(f)

    anns_by_image = defaultdict(list)
    for ann in data['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    candidate_ids = [img_id for img_id, anns in anns_by_image.items() if 1 <= len(anns) <= 4]

    random.seed(seed)
    random.shuffle(candidate_ids)

    selected_ids, category_seen = [], defaultdict(int)
    for img_id in candidate_ids:
        if len(selected_ids) >= target_size:
            break
        cats = set(a['category_id'] for a in anns_by_image[img_id])
        if any(category_seen[c] < max_per_category for c in cats):
            selected_ids.append(img_id)
            for c in cats:
                category_seen[c] += 1

    return selected_ids


def main():
    print("Step 1: Selecting balanced image subset...")
    selected_ids = select_balanced_subset(ANNOTATION_JSON, TARGET_SIZE, MAX_PER_CATEGORY)
    print(f"Selected {len(selected_ids)} images")

    print("\nStep 2: Building structured attribute metadata...")
    image_metadata = build_image_metadata(ANNOTATION_JSON, IMAGES_DIR, selected_ids)
    print(f"Tagged {len(image_metadata)} images")

    print("\nStep 3: Loading FashionCLIP and generating embeddings...")
    model, processor, device = load_fashion_clip()
    embeddings, valid_ids = embed_images(image_metadata, IMAGES_DIR, model, processor, device)
    print(f"Generated embeddings: {embeddings.shape}")

    print("\nStep 4: Building FAISS index...")
    index = build_faiss_index(embeddings)
    print(f"Indexed {index.ntotal} vectors")

    print("\nStep 5: Running evaluation queries...")
    run_evaluation(index, valid_ids, image_metadata, model, processor, device)


if __name__ == '__main__':
    main()
