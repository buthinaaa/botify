import logging
import time
import traceback
from api.services.nlp_manager import NLPManager

# Configure logger
logger = logging.getLogger(__name__)

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
    logger.warning("hybrid_search started")
    logger.debug(f"Query: '{query[:100]}...'")
    logger.debug(f"Query label: {query_label}")
    logger.debug(f"Parameters: top_n={top_n}, alpha={alpha}, beta={beta}, gamma={gamma}")
    logger.debug(f"Similarity threshold: {similarity_threshold}")
    logger.warning(f"Input data: {len(raw_chunks)} chunks, {len(chunk_labels)} labels")
    
    start_time = time.time()
    
    try:
        import numpy as np
        from nltk.tokenize import word_tokenize
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        # Validate inputs
        if not raw_chunks:
            logger.warning("No raw chunks provided")
            return []
        
        if faiss_index is None:
            logger.warning("FAISS index is None")
            return []
        
        if bm25_index is None:
            logger.warning("BM25 index is None")
            return []

        # Step 1: Encode query once
        logger.warning("STEP 1: Encoding query")
        encode_start = time.time()
        query_embedding = NLPManager.get_instance().embedding_model.encode([query])[0]
        encode_end = time.time()
        logger.warning(f"Query encoding completed in {encode_end - encode_start:.3f}s")
        logger.debug(f"Query embedding shape: {query_embedding.shape}")

        # Step 2: FAISS scores
        logger.warning("STEP 2: Computing FAISS scores")
        faiss_start = time.time()
        logger.debug(f"FAISS index type: {type(faiss_index)}")
        
        try:
            _, faiss_indices = faiss_index.search(np.array([query_embedding]), len(raw_chunks))
            faiss_ranks = faiss_indices[0]
            faiss_scores = np.linspace(1, 0, len(faiss_ranks))  # Highest score = 1
            faiss_end = time.time()
            logger.warning(f"FAISS search completed in {faiss_end - faiss_start:.3f}s")
            logger.debug(f"FAISS top 5 indices: {faiss_ranks[:5]}")
        except Exception as faiss_error:
            logger.error(f"FAISS search failed: {str(faiss_error)}")
            faiss_ranks = np.arange(len(raw_chunks))
            faiss_scores = np.zeros(len(raw_chunks))

        # Step 3: BM25 scores
        logger.warning("STEP 3: Computing BM25 scores")
        bm25_start = time.time()
        try:
            tokenized_query = word_tokenize(query.lower())
            logger.debug(f"Tokenized query: {tokenized_query}")
            
            bm25_scores = bm25_index.get_scores(tokenized_query)
            bm25_ranks = np.argsort(bm25_scores)[::-1]
            bm25_scores_sorted = np.sort(bm25_scores)[::-1]
            bm25_scores_sorted = bm25_scores_sorted / (np.max(bm25_scores_sorted) + 1e-9)  # Normalize
            bm25_end = time.time()
            logger.warning(f"BM25 search completed in {bm25_end - bm25_start:.3f}s")
            logger.debug(f"BM25 max score: {np.max(bm25_scores):.3f}")
            logger.debug(f"BM25 top 5 scores: {bm25_scores_sorted[:5]}")
        except Exception as bm25_error:
            logger.error(f"BM25 search failed: {str(bm25_error)}")
            bm25_scores = np.zeros(len(raw_chunks))
            bm25_ranks = np.arange(len(raw_chunks))

        # Step 4: Label similarity (semantic)
        logger.warning("STEP 4: Computing label similarity")
        label_start = time.time()
        try:
            query_label_embedding = NLPManager.get_instance().embedding_model.encode([query_label])[0]
            label_sim_scores = []
            
            for i, label in enumerate(chunk_labels):
                try:
                    chunk_label_embedding = NLPManager.get_instance().embedding_model.encode([label])[0]
                    sim = cosine_similarity(
                        [query_label_embedding], [chunk_label_embedding]
                    )[0][0]
                    final_sim = sim if sim >= similarity_threshold else 0
                    label_sim_scores.append(final_sim)
                    
                    if final_sim > 0:
                        logger.debug(f"Label {i}: '{label}' -> similarity: {sim:.3f}")
                except Exception as label_error:
                    logger.warning(f"Failed to compute similarity for label {i}: {str(label_error)}")
                    label_sim_scores.append(0)
            
            label_end = time.time()
            logger.warning(f"Label similarity computation completed in {label_end - label_start:.3f}s")
            logger.warning(f"Labels above threshold: {sum(1 for score in label_sim_scores if score > 0)}")
        except Exception as label_error:
            logger.error(f"Label similarity computation failed: {str(label_error)}")
            label_sim_scores = [0] * len(raw_chunks)

        # Step 5: Combine scores
        logger.warning("STEP 5: Combining scores")
        combine_start = time.time()
        final_scores = []
        
        for i in range(len(raw_chunks)):
            # FAISS score (rank-based)
            faiss_score = 1 - (np.where(faiss_ranks == i)[0][0] / len(faiss_ranks)) if i in faiss_ranks else 0
            
            # BM25 score (normalized)
            bm25_score = bm25_scores[i] / (np.max(bm25_scores) + 1e-9)
            
            # Label score
            label_score = label_sim_scores[i] if i < len(label_sim_scores) else 0

            combined_score = (
                alpha * faiss_score +
                beta * bm25_score +
                gamma * label_score
            )
            
            final_scores.append((i, combined_score))
            
            if i < 5:  # Log first 5 for debugging
                logger.debug(f"Chunk {i}: FAISS={faiss_score:.3f}, BM25={bm25_score:.3f}, Label={label_score:.3f} -> Combined={combined_score:.3f}")

        combine_end = time.time()
        logger.warning(f"Score combination completed in {combine_end - combine_start:.3f}s")

        # Step 6: Sort by combined score and return top_n chunks
        logger.warning("STEP 6: Sorting and selecting top chunks")
        sort_start = time.time()
        final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i for i, score in final_scores[:top_n]]
        top_chunks = [raw_chunks[i] for i in top_indices]
        sort_end = time.time()
        logger.warning(f"Sorting and selection completed in {sort_end - sort_start:.3f}s")

        end_time = time.time()
        total_time = end_time - start_time
        logger.warning(f"hybrid_search completed in {total_time:.3f}s")
        logger.warning(f"Returning {len(top_chunks)} chunks out of {len(raw_chunks)} total")
        
        # Log top results
        for i, (chunk_idx, score) in enumerate(final_scores[:top_n]):
            chunk_preview = raw_chunks[chunk_idx][:100] + "..." if len(raw_chunks[chunk_idx]) > 100 else raw_chunks[chunk_idx]
            logger.warning(f"Top {i+1}: Chunk {chunk_idx} (score: {score:.3f}) - '{chunk_preview}'")

        return top_chunks
        
    except Exception as e:
        end_time = time.time()
        logger.error(f"hybrid_search failed after {end_time - start_time:.3f}s: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty list on error
        return []
