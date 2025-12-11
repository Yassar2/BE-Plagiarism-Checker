# utils/helpers.py

def format_results(result, df):
    formatted = []

    for idx in result["top_indices"]:
        doc_idx = result["candidate_indices"][idx]

        formatted.append({
            "index": int(doc_idx),
            "title": str(df.loc[doc_idx].get("title", "")),
            "url": str(df.loc[doc_idx].get("url", "")),
            "pdf": str(df.loc[doc_idx].get("pdf", "")),
            "abstract": str(df.loc[doc_idx]["abstract"]),
            "lexical": float(result["lexical"][idx]),
            "semantic": float(result["semantic"][idx]),
            "structure": float(result["structure"][idx]),
            "final_score": float(result["final"][idx]),
            "similarity": float(result["final"][idx]) * 100
        })

    return formatted
