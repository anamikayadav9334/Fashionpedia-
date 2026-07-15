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
