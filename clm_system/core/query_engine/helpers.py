def reciprocal_rank_fusion(
    results_a: List[Dict],
    results_b: List[Dict],
    k: int = 60,
    weight_a: float = 1.0,
    weight_b: float = 1.0
) -> List[Dict]:
    """
    Combines search results using Reciprocal Rank Fusion algorithm.
    """
    fused_results = {}
    
    # Process first result list
    for idx, item in enumerate(results_a):
        doc_id = f"{item['contract_id']}_{item['clause_id']}"
        rank = idx + 1
        score = weight_a * (1 / (k + rank))
        fused_results[doc_id] = {
            **item,
            "rrf_score": score,
            "origin": "elastic"
        }
    
    # Process second result list and update scores
    for idx, item in enumerate(results_b):
        doc_id = f"{item['contract_id']}_{item['clause_id']}"
        rank = idx + 1
        score = weight_b * (1 / (k + rank))
        
        if doc_id in fused_results:
            # Document exists in both lists, merge and update score
            fused_results[doc_id]["rrf_score"] += score
            fused_results[doc_id]["origin"] = "both"
            # Keep metadata from both if they differ
            if item.get("metadata") != fused_results[doc_id].get("metadata"):
                fused_results[doc_id]["metadata"] = {
                    **fused_results[doc_id].get("metadata", {}),
                    **item.get("metadata", {})
                }
        else:
            # New document, add to results
            fused_results[doc_id] = {
                **item,
                "rrf_score": score,
                "origin": "vector"
            }
    
    # Convert back to list and sort by RRF score
    combined = list(fused_results.values())
    sorted_results = sorted(combined, key=lambda x: -x["rrf_score"])
    
    # Normalize scores to 0-1 range
    if sorted_results:
        max_score = max(r["rrf_score"] for r in sorted_results)
        if max_score > 0:
            for r in sorted_results:
                r["relevance_score"] = r["rrf_score"] / max_score
                del r["rrf_score"]
                del r["origin"]  # Clean up temp field
    
    return sorted_results