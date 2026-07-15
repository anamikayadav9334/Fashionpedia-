"""
Hybrid search: combines FashionCLIP dense embedding similarity (for
semantic/style/context understanding) with structured attribute matching
(for compositional precision - e.g. correctly binding "red" to "shirt"
rather than "pants"). This addresses CLIP's known weakness at binding
attributes to the correct object in multi-attribute queries.
"""
import sys
sys.path.append('..')
from indexer.embed_images import embed_text_query


def dense_search(query_text, index, faiss_id_order, model, processor, device, top_k=20):
    """Stage 1: FashionCLIP embedding similarity search via FAISS."""
    query_vec = embed_text_query(query_text, model, processor, device)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        img_id = faiss_id_order[idx]
        results.append({'image_id': img_id, 'dense_score': float(score)})
    return results


def attribute_match_score(query_text, img_id, image_metadata):
    """
    Stage 2: checks how well an image's structured tags (garment category +
    color) match words in the query. Rewards images where BOTH a garment
    type AND its color are mentioned together in the query, which is what
    solves the compositional binding problem plain CLIP struggles with.
    """
    query_lower = query_text.lower()
    garments = image_metadata[img_id]['garments']

    score = 0.0
    for g in garments:
        cat_words = g['category'].lower().replace(',', ' ').split()
        color_word = (g['color'] or '').lower()

        cat_hit = any(w in query_lower for w in cat_words if len(w) > 2)
        color_hit = color_word and color_word in query_lower

        if cat_hit and color_hit:
            score += 1.0
        elif cat_hit or color_hit:
            score += 0.3

    return score


def hybrid_search(query_text, index, faiss_id_order, image_metadata,
                   model, processor, device, top_k=5, dense_candidates=30, alpha=0.6):
    """
    Main retrieval function. Retrieves dense_candidates via embedding search,
    reranks using structured attribute matching, returns top_k.

    alpha: weight on dense score vs attribute score (0.6 = 60% dense, 40% attribute)
    """
    dense_results = dense_search(query_text, index, faiss_id_order, model, processor, device, top_k=dense_candidates)

    for r in dense_results:
        attr_score = attribute_match_score(query_text, r['image_id'], image_metadata)
        r['attr_score'] = attr_score
        r['final_score'] = alpha * r['dense_score'] + (1 - alpha) * attr_score

    dense_results.sort(key=lambda x: -x['final_score'])
    return dense_results[:top_k]
