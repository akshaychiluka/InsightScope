import re
import os 
import numpy as np
import faiss
from textblob import TextBlob
from collections import Counter
import google.generativeai as genai
from langchain_community.document_loaders import UnstructuredURLLoader
import streamlit as st
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure API key
genai.configure(api_key="AIzaSyD7StEDoYZCCJaBXq1aK8CyMtcuetnb22I")

def is_valid_url(url):
    """Check if a URL is valid."""
    return re.match(r'^https?://', url) is not None

def extract_text_from_url(url):
    """Extract text from a URL."""
    try:
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()
        if docs and docs[0].page_content:
            return docs[0].page_content
        else:
            st.warning(f"⚠️ No content extracted from {url}")
            return ""
    except Exception as e:
        st.error(f"❌ Error extracting text from {url}: {e}")
        return ""

def get_embedding(text):
    """Get the embedding of a text."""
    try:
        response = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")
        return response["embedding"] if "embedding" in response else None
    except Exception as e:
        st.error(f"❌ Error generating embedding: {e}")
        return None

def store_embeddings(texts, urls):
    """Store embeddings and metadata."""
    embeddings = []
    for text, url in zip(texts, urls):
        if text:
            embedding = get_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                st.warning(f"⚠️ No embedding generated for content from {url}")
        else:
            st.warning(f"⚠️ No content extracted from {url}")

    if not embeddings:
        st.error("❌ No valid embeddings generated.")
        return

    user_id = st.session_state.user['localId']
    user_faiss_path = f"faiss_store_{user_id}"
    user_metadata_path = f"metadata_{user_id}.pkl"
    
    vector_np = np.array(embeddings, dtype=np.float32)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(vector_np)
    faiss.write_index(index, user_faiss_path)
    
    metadata = list(zip(texts, urls))
    with open(user_metadata_path, "wb") as f:
        import pickle
        pickle.dump(metadata, f)

def generate_brief(text):
    """Generate a brief summary of a text."""
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    try:
        response = model.generate_content(f"Provide a brief summary of the following text in 2-3 sentences:\n\n{text}")
        return response.text if response else "Unable to generate brief."
    except Exception as e:
        logger.error(f"Error generating brief: {e}")
        return "Unable to generate brief due to an error."

def generate_question_recommendations(texts, last_query=None):
    """Generate question recommendations based on texts and last query."""
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    combined_brief = "\n\n".join([f"URL {i+1} Summary:\n{generate_brief(text)}" for i, text in enumerate(texts)])
    prompt = f"Based on the summaries below and last query '{last_query}'" if last_query else f"Based on the summaries below"
    prompt += f", generate exactly 5 specific questions:\n\n{combined_brief}"
    
    try:
        response = model.generate_content(prompt)
        questions = [re.sub(r"^\d+\.\s*|- ", "", line.strip()) for line in response.text.split("\n") 
                    if line.strip().endswith("?") and len(line) > 10][:5] if response else []
        return questions or [
            "What are the key financial highlights across all articles?",
            "How do recent market trends affect stocks in these URLs?",
            "Which companies are mentioned across the articles?",
            "What are the latest stock prices reported in these URLs?",
            "Can you summarize the market outlook from all articles?"
        ]
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return [
            "What are the key financial highlights across all articles?",
            "How do recent market trends affect stocks in these URLs?",
            "Which companies are mentioned across the articles?",
            "What are the latest stock prices reported in these URLs?",
            "Can you summarize the market outlook from all articles?"
        ]

def analyze_sentiment(text):
    """Analyze the sentiment of a text."""
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return ("Positive", sentiment_score) if sentiment_score > 0 else ("Negative", sentiment_score) if sentiment_score < 0 else ("Neutral", sentiment_score)

def retrieve_texts_and_urls(query, top_k=1, filter_option="Relevance"):
    """Retrieve texts and URLs based on a query."""
    user_id = st.session_state.user['localId']
    user_faiss_path = f"faiss_store_{user_id}"
    user_metadata_path = f"metadata_{user_id}.pkl"
    
    if os.path.exists(user_faiss_path) and os.path.exists(user_metadata_path):
        index = faiss.read_index(user_faiss_path)
        with open(user_metadata_path, "rb") as f:
            import pickle
            metadata = pickle.load(f)
        
        texts, urls = zip(*metadata)
        url_match = re.search(r"url\s*(\d+)", query.lower())
        if url_match:
            url_index = int(url_match.group(1)) - 1
            return [texts[url_index]], [urls[url_index]] if 0 <= url_index < len(texts) else (None, None)
        
        query_embedding = np.array([get_embedding(query)], dtype=np.float32)
        distances, indices = index.search(query_embedding, k=top_k)
        retrieved_texts = [texts[i] for i in indices[0] if i < len(texts)]
        retrieved_urls = [urls[i] for i in indices[0] if i < len(urls)]

        if filter_option == "Keyword Frequency":
            query_words = set(query.lower().split())
            scored_texts = [(sum(text.lower().count(word) for word in query_words), text, url) 
                          for text, url in zip(retrieved_texts, retrieved_urls)]
            scored_texts.sort(reverse=True)
            retrieved_texts = [text for _, text, _ in scored_texts[:top_k]]
            retrieved_urls = [url for _, _, url in scored_texts[:top_k]]

        return retrieved_texts, retrieved_urls
    return None, None

def generate_answer(query, context, url):
    """Generate an answer based on a query, context, and URL."""
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    try:
        response = model.generate_content(f"Context:\n{context}\n\nUser: {query}\n\nAssistant:")
        return response.text if response else "I couldn’t find an answer.", url
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Unable to generate answer due to an error.", url