# Multimodal Fashion & Context Retrieval

A search engine that retrieves fashion images from natural language descriptions,
understanding garment attributes ("what"), scene/context ("where"), and style ("vibe").

Built for the Glance ML Internship Assignment.

## Approach

Vanilla CLIP-style retrieval treats an image as one global embedding, which makes it
struggle with compositional queries (e.g. distinguishing "red shirt, blue pants" from
"blue shirt, red pants"). This project uses a **hybrid** approach:

1. **Dense retrieval** via [FashionCLIP](https://github.com/patrickjohncyh/fashion-clip)
   embeddings (fashion-domain fine-tuned CLIP) for semantic/style/context understanding.
2. **Structured attribute matching** — each image is tagged with per-garment category
   and color (extracted via bounding-box color sampling on
   [Fashionpedia](https://fashionpedia.github.io/home/) annotations), used to rerank
   dense results and enforce compositional precision.

Final score = weighted combination of dense similarity + attribute match score.

## Repo structure
fashion-retrieval/
├── indexer/
│   ├── embed_images.py       # FashionCLIP embedding extraction
│   ├── extract_attributes.py # parse Fashionpedia annotations → structured tags
│   └── build_index.py        # push embeddings + metadata into Chroma
├── retriever/
│   ├── parse_query.py        # NL query → (color, garment, scene) tags
│   ├── search.py             # dense search + attribute rerank
│   └── evaluate.py           # run the 5 eval queries, print top-k results
├── data/                     # (gitignored — don't commit raw images)
├── requirements.txt
├── README.md
└── writeup.pdf
## Dataset

[Fashionpedia](https://fashionpedia.github.io/home/) — chosen over generic product-catalog
datasets (e.g. plain-background e-commerce photos) because it contains real-world images
with actual environments/backgrounds (street, indoor, etc.), which is required for
context-aware queries like "professional attire in a modern office." A balanced subset of
~800-820 images was selected, capping overrepresented garment categories so rarer types
still get fair representation.

## Setup & running

```bash
pip install torch transformers faiss-cpu pillow numpy
python run_pipeline.py
```

Update the `ANNOTATION_JSON` and `IMAGES_DIR` paths at the top of `run_pipeline.py` to
point to your local Fashionpedia download.

## Known limitations

- Fashionpedia has no native color labels; colors are extracted via median-pixel sampling
  within each garment's bounding box, mapped to the nearest named color in a fixed palette.
  This is a reasonable approximation but not as precise as a dedicated color classifier.
- Some rare garment categories (e.g. ties) are underrepresented in a random/balanced
  subset of this size; a targeted top-up step was used to ensure evaluation query coverage.
- The dataset's images are lifestyle/editorial photography, not natural everyday photos,
  which may not perfectly match casual-context queries (e.g. "city walk").

## Future work

See the accompanying write-up PDF for full details on extending this system with
location/weather metadata and further precision improvements.
