# InsightScope: AI-Powered News Research Tool

InsightScope is an AI-driven web application designed to simplify news research. It allows users to extract content from news article URLs, ask questions, and receive intelligent answers with sentiment analysis and dynamic question suggestions.

Built with **Streamlit**, **Firebase**, **FAISS**, and **Gemini API**, it’s ideal for students, researchers, and professionals seeking quick insights from news articles.

---

## ✨ Features

- **URL Content Extraction**: Process up to 10 news article URLs to extract and index text.
- **AI Q&A**: Ask questions and receive concise AI-powered answers via Gemini API.
- **Sentiment Analysis**: Evaluate article sentiment (positive, negative, neutral) using TextBlob.
- **Dynamic Suggestions**: Receive 5 AI-generated follow-up questions, with a reload option.
- **Data Visualization**: Explore word frequency through interactive Plotly charts.
- **Export Options**: Save your chat history as CSV or PDF.
- **Secure Authentication**: Firebase-based sign-up, sign-in, and email verification.
- **Modern UI**: Dark-themed, responsive interface with custom CSS.

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, Firebase (Authentication), FAISS (Vector Search)
- **AI/ML**: Gemini API (text embedding and answer generation), TextBlob (sentiment analysis)
- **Visualization**: Plotly, Pandas
- **Other Libraries**: LangChain, ReportLab, Pyrebase

---

## ⚙️ Prerequisites

- Python 3.8+
- Firebase Project with **Email/Password Authentication** enabled
- Gemini API Key from Google
- FAISS (CPU version recommended)

---
