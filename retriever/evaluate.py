"""
Runs the 5 required evaluation queries from the assignment and prints
top-5 results with scores for manual/visual inspection.
"""
import sys
sys.path.append('..')
from retriever.search import hybrid_search

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def run_evaluation(index, faiss_id_order, image_metadata, model, processor, device, top_k=5):
    """Runs all 5 evaluation queries and returns a dict of query -> results."""
    all_results = {}
    for query in EVAL_QUERIES:
        results = hybrid_search(query, index, faiss_id_order, image_metadata,
                                 model, processor, device, top_k=top_k)
        all_results[query] = results

        print(f"\n{'='*80}\nQUERY: {query}\n{'='*80}")
        for r in results:
            img_id = r['image_id']
            cats = set(g['category'] for g in image_metadata[img_id]['garments'])
            colors = set(g['color'] for g in image_metadata[img_id]['garments'] if g['color'])
            print(f"  id={img_id} final={r['final_score']:.3f} "
                  f"dense={r['dense_score']:.3f} attr={r['attr_score']:.2f} | {cats} | {colors}")

    return all_results
