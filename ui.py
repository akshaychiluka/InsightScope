import streamlit as st
import plotly.express as px
from utils import (
    is_valid_url,
    extract_text_from_url,
    get_embedding,
    store_embeddings,
    generate_brief,
    generate_question_recommendations,
    analyze_sentiment,
    retrieve_texts_and_urls,
    generate_answer,
    Counter
)
from auth import sign_out
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_word_frequency_visualization(texts):
    combined_text = " ".join(texts).lower()
    words = combined_text.split()
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    word_counts = Counter(filtered_words).most_common(10)
    
    df = pd.DataFrame(word_counts, columns=["Word", "Frequency"])
    fig = px.bar(df, x="Frequency", y="Word", orientation="h", title="Top 10 Frequent Words",
                 color="Frequency", color_continuous_scale="Viridis", height=400, width=600)
    fig.update_layout(title_font_size=16, title_font_color="#e0e0ff", font=dict(size=12, color="#e0e0ff"),
                      plot_bgcolor="black", paper_bgcolor="black")
    return fig

def export_to_csv(history):
    data = [{"Query": entry["query"], "Answer": entry["answer"], "Source URL": entry["source"]} for entry in history]
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def export_to_pdf(history):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.textColor = "#333333"
    style.fontName = "Helvetica"
    elements = []

    elements.append(Paragraph("InsightScope Chat History", styles["Title"]))
    elements.append(Spacer(1, 12))

    for entry in history:
        elements.append(Paragraph(f"<b>Query:</b> {entry['query']}", style))
        elements.append(Paragraph(f"<b>Answer:</b> {entry['answer']}", style))
        elements.append(Paragraph(f"<b>Source URL:</b> {entry['source']}", style))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

def main_app():
    st.title("📈 InsightScope: AI-Powered News Research Tool")
    st.markdown("**Chat with your news articles, explore insights, and get dynamic AI-generated question suggestions!**")
    
    # Sidebar
    st.sidebar.title("🔗 News Article URLs")
    st.sidebar.write(f"Welcome, {st.session_state.user['email']}")
    if st.sidebar.button("Sign Out"):
        sign_out()
        st.rerun()
    
    num_urls = st.sidebar.number_input("Number of URLs", min_value=1, max_value=10, value=3)
    urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(num_urls)]
    process_url_clicked = st.sidebar.button("🚀 Process URLs")
    filter_option = st.sidebar.selectbox("Filter Results By:", ["Relevance", "Keyword Frequency"])
    show_dashboard = st.sidebar.checkbox("Show Visualization Dashboard", value=False)
    
    if process_url_clicked:
        st.session_state.history = []
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about the news articles you’ve processed."}]
        st.session_state.extracted_texts = []
        st.session_state.suggested_questions = []
        st.session_state.url_list = []
        st.session_state.last_query = None

        with st.spinner("🔍 Extracting text from URLs..."):
            valid_urls_input = [url for url in urls if url and is_valid_url(url)]
            if not valid_urls_input:
                st.error("❌ No valid URLs provided.")
            else:
                progress_bar = st.progress(0)
                extracted_texts = []
                valid_urls = []
                for i, url in enumerate(valid_urls_input):
                    extracted_text = extract_text_from_url(url)
                    if extracted_text:
                        extracted_texts.append(extracted_text)
                        valid_urls.append(url)
                    progress_bar.progress((i + 1) / len(valid_urls_input))
                if extracted_texts:
                    store_embeddings(extracted_texts, valid_urls)
                    st.session_state.extracted_texts = extracted_texts
                    st.session_state.url_list = valid_urls
                    with st.spinner("Generating question suggestions..."):
                        st.session_state.suggested_questions = generate_question_recommendations(extracted_texts)
                    st.success(f"✅ Processed {len(valid_urls)} URL(s).")
    
    if show_dashboard and st.session_state.extracted_texts:
        st.sidebar.subheader("📊 Visualization Dashboard")
        fig = generate_word_frequency_visualization(st.session_state.extracted_texts)
        st.sidebar.plotly_chart(fig, use_container_width=True)
    
    # Chat Interface
    st.subheader("💬 Chat with InsightScope")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(f"<div class='answer-box'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Suggested Questions
    if st.session_state.suggested_questions:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("**Suggested Questions:**")
        with col2:
            if st.button("🔄 Reload Suggestions", key="reload_suggestions"):
                with st.spinner("Generating new suggestions..."):
                    st.session_state.suggested_questions = generate_question_recommendations(st.session_state.extracted_texts, st.session_state.last_query)
        
        question_cols = st.columns(5)
        for i, question in enumerate(st.session_state.suggested_questions):
            with question_cols[i]:
                if st.button(question, key=f"suggest_{i}"):
                    prompt = question
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(f"<div class='answer-box'>{prompt}</div>", unsafe_allow_html=True)
                    retrieved_texts, retrieved_urls = retrieve_texts_and_urls(prompt, filter_option=filter_option)
                    if retrieved_texts:
                        answer, source_url = generate_answer(prompt, retrieved_texts[0], retrieved_urls[0])
                        sentiment_label, sentiment_score = analyze_sentiment(retrieved_texts[0])
                        response = f"{answer}\n\n**Sentiment:** {sentiment_label} (Score: {sentiment_score:.2f})\n**Source:** [{source_url}]({source_url})"
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with chat_container:
                            with st.chat_message("assistant"):
                                st.markdown(f"<div class='answer-box'>{response}</div>", unsafe_allow_html=True)
                        st.session_state.history.append({"query": prompt, "answer": answer, "source": source_url})
                        if len(st.session_state.history) > 5:
                            st.session_state.history.pop(0)
                        st.session_state.last_query = prompt
                        st.session_state.suggested_questions = generate_question_recommendations(st.session_state.extracted_texts, prompt)
    
    # Chat Input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(f"<div class='answer-box'>{prompt}</div>", unsafe_allow_html=True)
        retrieved_texts, retrieved_urls = retrieve_texts_and_urls(prompt, filter_option=filter_option)
        if retrieved_texts:
            answer, source_url = generate_answer(prompt, retrieved_texts[0], retrieved_urls[0])
            sentiment_label, sentiment_score = analyze_sentiment(retrieved_texts[0])
            response = f"{answer}\n\n**Sentiment:** {sentiment_label} (Score: {sentiment_score:.2f})\n**Source:** [{source_url}]({source_url})"
            st.session_state.messages.append({"role": "assistant", "content": response})
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(f"<div class='answer-box'>{response}</div>", unsafe_allow_html=True)
            st.session_state.history.append({"query": prompt, "answer": answer, "source": source_url})
            if len(st.session_state.history) > 5:
                st.session_state.history.pop(0)
            st.session_state.last_query = prompt
            st.session_state.suggested_questions = generate_question_recommendations(st.session_state.extracted_texts, prompt)
    
    # History
    if st.session_state.history:
        st.subheader("📜 Recent Query History (Last 5)")
        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Query {len(st.session_state.history) - i}: {entry['query']}"):
                st.markdown(f"<div class='history-box'><strong>Answer:</strong> {entry['answer']}<br><strong>Source:</strong> <a href='{entry['source']}' target='_blank'>{entry['source']}</a></div>", unsafe_allow_html=True)
    
    # Export
    if st.session_state.history:
        st.sidebar.subheader("📥 Export Chat History")
        export_format = st.sidebar.selectbox("Choose export format:", ["CSV", "PDF"])
        data = export_to_csv(st.session_state.history) if export_format == "CSV" else export_to_pdf(st.session_state.history)
        file_name = f"insightscope_chat_history.{export_format.lower()}"
        mime = "text/csv" if export_format == "CSV" else "application/pdf"
        st.sidebar.download_button(label=f"Download as {export_format}", data=data, file_name=file_name, mime=mime)