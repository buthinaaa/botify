import os
import fitz
from docx import Document
import nltk
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from api.services.nlp_manager import NLPManager

def clean_text(text):
    """Cleans text by removing special characters, multiple spaces, and numbers."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()

def preprocess_text(text):
    """Tokenizes, removes stopwords, and lemmatizes text."""
    tokens = word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatized_tokens = [NLPManager.get_instance().nlp(word)[0].lemma_ for word in filtered_tokens]
    return " ".join(lemmatized_tokens)  # Return processed text


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return clean_text(text)


def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return clean_text(text)


def extract_text_from_txt(txt_path):
    """Extracts text from a TXT file."""
    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read()
    return clean_text(text)


def create_bm25_index(documents):
    """Creates a BM25 index from tokenized documents."""
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents if doc.strip()]
    
    # Filter out empty tokenized docs
    tokenized_docs = [doc for doc in tokenized_docs if len(doc) > 0]
    
    if not tokenized_docs:
        raise ValueError("No valid documents to create BM25 index.")

    return BM25Okapi(tokenized_docs), tokenized_docs



def create_faiss_index(documents):
    """Creates a FAISS index from embedded document vectors."""
    document_embeddings = NLPManager.get_instance().embedding_model.encode(documents)
    dimension = document_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(document_embeddings))
    return faiss_index, document_embeddings

from sklearn.metrics.pairwise import cosine_similarity

def semantic_window_chunking(text, window_size=3, stride=1, similarity_threshold=0.7, return_embeddings=False):
    """
    Chunks text into semantically meaningful segments using sentence embeddings.

    Parameters:
    - window_size: Number of sentences per chunk.
    - stride: Number of sentences to slide the window.
    - similarity_threshold: Threshold for merging similar chunks.
    - return_embeddings: If True, also returns the embeddings of the merged chunks.

    Returns:
    - merged_chunks: List of semantically grouped text chunks.
    - chunk_embeddings (optional): List of embeddings corresponding to merged chunks.
    """
    # Step 1: Split text into sentences
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= window_size:
        if return_embeddings:
            return [text], NLPManager.get_instance().embedding_model.encode([text])
        return [text]

    # Step 2: Compute embeddings for each sentence
    sentence_embeddings = NLPManager.get_instance().embedding_model.encode(sentences)

    # Step 3: Slide a window of size `window_size`
    chunks = []
    i = 0
    while i + window_size <= len(sentences):
        window_sentences = sentences[i:i + window_size]
        window_text = " ".join(window_sentences)
        chunks.append(window_text)
        i += stride

    # Step 4: Merge similar consecutive chunks
    merged_chunks = [chunks[0]]
    for j in range(1, len(chunks)):
        sim = cosine_similarity(
            NLPManager.get_instance().embedding_model.encode([chunks[j - 1]]),
            NLPManager.get_instance().embedding_model.encode([chunks[j]])
        )[0][0]
        if sim > similarity_threshold:
            merged_chunks[-1] += " " + chunks[j]
        else:
            merged_chunks.append(chunks[j])

    if return_embeddings:
        chunk_embeddings = NLPManager.get_instance().embedding_model.encode(merged_chunks)
        return merged_chunks, chunk_embeddings

    return merged_chunks

def generate_labels_keybert(chunks, top_n=1, ngram_range=(1, 3)):
    """
    Generate one semantic label per chunk using KeyBERT.

    Parameters:
    - chunks: list of str, the text chunks to label
    - top_n: int, how many top keywords to consider (default=1)
    - ngram_range: tuple, the range of n-grams to consider (default=(1, 3))

    Returns:
    - labels: list of str, one label per chunk
    """
    labels = []
    for chunk in chunks:
        keywords = NLPManager.get_instance().kw_model.extract_keywords(
            chunk,
            keyphrase_ngram_range=ngram_range,
            stop_words='english',
            top_n=top_n
        )
        label = keywords[0][0].replace(" ", "_") if keywords else "No_Label"
        labels.append(label)
    return labels


def preprocessing_docs(file_paths):
    """
    Loads documents from multiple file types (PDF, TXT, DOCX), preprocesses them, and creates BM25 and FAISS indexes.

    - file_paths: List of file paths (PDF, TXT, or DOCX) to be processed.

    Returns:
    - raw_chunks: List of semantic chunks (raw) for embedding/QA.
    - bm25_index: BM25 index (using preprocessed chunks).
    - tokenized_docs: Tokenized documents for BM25.
    - faiss_index: FAISS index (using raw chunks).
    - document_embeddings: Embeddings of raw chunks.
    - intent_labels: Extracted topic/intent labels using NMF or LDA.
    """
    docs = []

    # Load and clean documents
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            docs.append(extract_text_from_pdf(file_path))
        elif file_path.endswith(".txt"):
            docs.append(extract_text_from_txt(file_path))
        elif file_path.endswith(".docx"):
            docs.append(extract_text_from_docx(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    # Semantic chunking (clean text) first
    raw_chunks = []
    all_embeddings = []
    for doc in docs:
        chunks, embeddings = semantic_window_chunking(doc, return_embeddings=True)
        raw_chunks.extend(chunks)
        all_embeddings.extend(embeddings)

    # Preprocess chunks only for keyword-based methods
    processed_chunks = [preprocess_text(chunk) for chunk in raw_chunks]

    # BM25 index on preprocessed data
    bm25_index, tokenized_docs = create_bm25_index(processed_chunks)

    # FAISS index on raw semantic chunks
    faiss_index, document_embeddings = create_faiss_index(raw_chunks)

    # Topic/intent labeling from preprocessed chunks
    intent_labels = generate_labels_keybert(raw_chunks)

    return raw_chunks, bm25_index, tokenized_docs, faiss_index, document_embeddings, intent_labels

def process_uploaded_files(file_paths):    
    raw_chunks, bm25_index, tokenized_docs, faiss_index,document_embeddings,intent_labels = preprocessing_docs(file_paths)
    if any(r is None or (isinstance(r, (list, str)) and len(r) == 0) for r in [raw_chunks, bm25_index, tokenized_docs, faiss_index, document_embeddings, intent_labels]):
        raise ValueError("One or more of the return values from preprocessing_docs are invalid.")
    print("Chunks:", raw_chunks)
    print("intent labels:", intent_labels)
    print("BM25 Index:", bm25_index)
    print("Tokenized Documents:", tokenized_docs)
    print("FAISS Index:", faiss_index)
    print("Document Embeddings:", document_embeddings)  
    return raw_chunks, bm25_index, tokenized_docs, faiss_index, document_embeddings, intent_labels