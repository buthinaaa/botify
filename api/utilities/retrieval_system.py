from api.services.nlp_manager import NLPManager


def hybrid_search(
    query,
    query_label,
    bm25_index,
    faiss_index,
    raw_chunks,
    document_embeddings,
    chunk_labels,
    top_n=5,
    alpha=0.5,
    beta=0.3,
    gamma=0.2,
    similarity_threshold=0.5
):
    """
    Hybrid search to retrieve top N most relevant chunks using FAISS, BM25, and label similarity.

    Parameters:
    - query: User query string
    - query_label: Label predicted for the query (from intent recognition model)
    - bm25_index: Precomputed BM25 index
    - faiss_index: Precomputed FAISS index
    - raw_chunks: List of raw text chunks (same order as used in indexes)
    - document_embeddings: Embeddings for raw_chunks (used in FAISS)
    - chunk_labels: List of labels for each chunk
    - top_n: Number of top chunks to return
    - alpha, beta, gamma: Weights for FAISS, BM25, and label similarity scores
    - similarity_threshold: Minimum semantic similarity between query label and chunk label

    Returns:
    - List of top_n most relevant raw chunks
    """
    import numpy as np
    from nltk.tokenize import word_tokenize
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    # Step 1: Encode query once
    query_embedding = NLPManager.get_instance().embedding_model.encode([query])[0]

    # Step 2: FAISS scores
    _, faiss_indices = faiss_index.search(np.array([query_embedding]), len(raw_chunks))
    faiss_ranks = faiss_indices[0]
    faiss_scores = np.linspace(1, 0, len(faiss_ranks))  # Highest score = 1

    # Step 3: BM25 scores
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranks = np.argsort(bm25_scores)[::-1]
    bm25_scores_sorted = np.sort(bm25_scores)[::-1]
    bm25_scores_sorted = bm25_scores_sorted / (np.max(bm25_scores_sorted) + 1e-9)  # Normalize

    # Step 4: Label similarity (semantic)
    query_label_embedding = NLPManager.get_instance().embedding_model.encode([query_label])[0]
    label_sim_scores = []
    for label in chunk_labels:
        chunk_label_embedding = NLPManager.get_instance().embedding_model.encode([label])[0]
        sim = cosine_similarity(
            [query_label_embedding], [chunk_label_embedding]
        )[0][0]
        label_sim_scores.append(sim if sim >= similarity_threshold else 0)

    # Step 5: Combine scores
    final_scores = []
    for i in range(len(raw_chunks)):
        faiss_score = 1 - (np.where(faiss_ranks == i)[0][0] / len(faiss_ranks)) if i in faiss_ranks else 0
        bm25_score = bm25_scores[i] / (np.max(bm25_scores) + 1e-9)
        label_score = label_sim_scores[i]

        combined_score = (
            alpha * faiss_score +
            beta * bm25_score +
            gamma * label_score
        )
        final_scores.append((i, combined_score))

    # Step 6: Sort by combined score and return top_n chunks
    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, score in final_scores[:top_n]]
    top_chunks = [raw_chunks[i] for i in top_indices]

    return top_chunks
