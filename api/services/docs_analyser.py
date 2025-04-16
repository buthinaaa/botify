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


def generate_intent_labels(processed_documents, num_keywords=2, model_type="LDA"):
    """
    Extracts intent labels from preprocessed documents.

    - processed_documents: List of cleaned and tokenized document texts.
    - num_keywords: Number of keywords per topic to form the intent labels.
    - model_type: Choose between "LDA" (Latent Dirichlet Allocation) or "NMF" (Non-Negative Matrix Factorization).

    Returns: A list of generated intent labels.
    """
    if not processed_documents:
        raise ValueError("No documents provided.")

    vectorizer = TfidfVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(processed_documents)

    # Dynamically determine the number of topics
    num_docs = len(processed_documents)
    avg_doc_length = sum(len(doc.split()) for doc in processed_documents) / num_docs
    vocab_size = len(vectorizer.get_feature_names_out())

    # Heuristic: Number of topics based on average document length and vocabulary size
    # Adjust these weights as needed
    num_topics = max(2, min(10, math.ceil(math.sqrt(avg_doc_length * vocab_size) / 10)))

    print(f"Number of topics determined dynamically: {num_topics}")

    if model_type == "LDA":
        model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    elif model_type == "NMF":
        model = NMF(n_components=num_topics, random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose 'LDA' or 'NMF'.")

    model.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        keywords = [feature_names[i] for i in topic.argsort()[:-num_keywords - 1:-1]]
        topics.append("_".join(keywords))

    return topics  # Returns a list of generated intent labels

def preprocessing_docs(file_paths):
    """
    Loads documents from multiple file types (PDF, TXT, DOCX), preprocesses them, and creates BM25 and FAISS indexes.

    - file_paths: List of file paths (PDF, TXT, or DOCX) to be processed.

    Returns:
    - processed_docs: List of preprocessed document texts.
    - bm25_index: BM25 index for text search.
    - tokenized_docs: Tokenized documents for BM25.
    - faiss_index: FAISS index for vector search.
    - document_embeddings: Embeddings of the documents.
    """
    docs = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            docs.append(extract_text_from_pdf(file_path))
        elif file_path.endswith(".txt"):
            docs.append(extract_text_from_txt(file_path))
        elif file_path.endswith(".docx"):
            docs.append(extract_text_from_docx(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    
    processed_docs = [preprocess_text(doc) for doc in docs]
    
    bm25_index, tokenized_docs = create_bm25_index(processed_docs)

    # Create FAISS index
    faiss_index, document_embeddings = create_faiss_index(processed_docs)

    # generating labels
    intent_labels = generate_intent_labels(processed_docs, model_type="NMF")
    if not intent_labels:
        raise ValueError("No intent labels generated.")
    if not tokenized_docs:
        raise ValueError("BM25 tokenized documents are empty.")
    return processed_docs, bm25_index, tokenized_docs, faiss_index, document_embeddings, intent_labels

def process_uploaded_files(file_paths):    
    processed_docs, bm25_index, tokenized_docs, faiss_index,document_embeddings,intent_labels = preprocessing_docs(file_paths)
    if any(r is None or (isinstance(r, (list, str)) and len(r) == 0) for r in [processed_docs, bm25_index, tokenized_docs, faiss_index, document_embeddings, intent_labels]):
        raise ValueError("One or more of the return values from preprocessing_docs are invalid.")
    print("Processed Documents:", processed_docs)
    print("intent labels:", intent_labels)
    print("BM25 Index:", bm25_index)
    print("Tokenized Documents:", tokenized_docs)
    print("FAISS Index:", faiss_index)
    print("Document Embeddings:", document_embeddings)  
    return processed_docs, bm25_index, tokenized_docs, faiss_index, document_embeddings, intent_labels