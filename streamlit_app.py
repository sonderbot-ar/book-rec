"""
The AI Librarian — Streamlit App (Starry Night Edition v2)
===========================================================
A book recommendation chatbot powered by RAG + LLM.

"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from librarian_chat import ask_librarian
import os


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="The AI Librarian",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
#  STARRY NIGHT THEME (CSS)
# ─────────────────────────────────────────────

STARRY_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Lora:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(
            180deg,
            #070e1a 0%, #0a1628 8%, #0d1b2a 20%,
            #0f2035 35%, #122744 50%, #0f2035 65%,
            #0d1b2a 80%, #0a1628 92%, #070e1a 100%
        );
        color: #e8dcc8;
        font-family: 'Lora', Georgia, serif;
    }

    /* ── Hide Streamlit Defaults ── */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0d1b2a 50%, #0a1628 100%) !important;
        border-right: 1px solid #1b2838;
    }
    [data-testid="stSidebar"] * {
        color: #e8dcc8 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #d4a843 !important;
    }

    /* ── Headers ── */
    .main-title {
        font-family: 'Playfair Display', Georgia, 'Times New Roman', serif;
        color: #e6b800;
        text-align: center;
        font-weight: 500;
        letter-spacing: 3px;
        font-size: 2.8em;
        text-shadow: 0 0 30px rgba(230, 184, 0, 0.25);
        padding: 20px 0 5px 0;
        margin: 0;
    }
    .main-subtitle {
        font-family: 'Lora', Georgia, serif;
        color: #4e6382;
        text-align: center;
        font-style: italic;
        font-size: 1.05em;
        letter-spacing: 0.5px;
        padding-bottom: 25px;
        margin: 0;
    }
    .section-header {
        font-family: 'Playfair Display', Georgia, serif;
        color: #d4a843;
        font-weight: 400;
        letter-spacing: 1.5px;
        font-size: 1.3em;
        margin: 20px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #1b2838;
    }

    /* ── Chat Messages ── */
    [data-testid="stChatMessage"] {
        font-family: 'Lora', Georgia, serif;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li {
        color: #e8dcc8 !important;
        font-family: 'Lora', Georgia, serif !important;
        line-height: 1.6;
    }
    
    /* ── Chat Input ── */
    [data-testid="stChatInput"] {
        border: 1px solid #d4a843 !important;
        border-radius: 10px !important;
        background: #0d1b2a !important;
        box-shadow: 0 0 10px rgba(212, 168, 67, 0.15) !important;
    }
    [data-testid="stChatInput"] > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #1b2838 !important;
        color: #f5f0e1 !important;
        border: 1px solid #2c5f8a !important;
        border-radius: 8px !important;
        font-family: 'Lora', Georgia, serif !important;
        font-size: 0.95em;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #d4a843 !important;
        box-shadow: 0 0 8px rgba(212, 168, 67, 0.3) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #4e6382 !important;
        font-style: italic;
    }
    [data-testid="stChatInput"] button {
        background: #d4a843 !important;
        color: #0a1628 !important;
        border: none !important;
        border-radius: 6px !important;
    }

    /* ── Chat Input Container Border ── */
    [data-testid="stChatInput"] {
        border: 1px solid #d4a843 !important;
        border-radius: 10px !important;
        background: #0d1b2a !important;
    }
    [data-testid="stChatInput"] > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    /* ── Book Card ── */
    .book-card {
        display: flex;
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
        border: 1px solid #2c5f8a;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        gap: 16px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .book-card:hover {
        border-color: #d4a843;
        box-shadow: 0 0 15px rgba(212, 168, 67, 0.15);
    }
    .book-cover {
        width: 100px;
        min-width: 100px;
        height: 150px;
        object-fit: cover;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .book-cover-placeholder {
        width: 100px;
        min-width: 100px;
        height: 150px;
        background: linear-gradient(135deg, #1b2838, #2c5f8a);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5em;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .book-info {
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 4px;
    }
    .book-title {
        font-family: 'Playfair Display', Georgia, serif;
        color: #e6b800;
        font-size: 1.15em;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin: 0;
        line-height: 1.3;
    }
    .book-author {
        font-family: 'Lora', Georgia, serif;
        color: #e8dcc8;
        font-size: 0.95em;
        font-style: italic;
        margin: 2px 0;
    }
    .book-meta {
        font-family: 'Lora', Georgia, serif;
        color: #8a9ab5;
        font-size: 0.85em;
        margin: 1px 0;
    }
    .book-rating {
        font-family: 'Lora', Georgia, serif;
        color: #d4a843;
        font-size: 0.9em;
        font-weight: 500;
        margin: 2px 0;
    }

    /* ── Sidebar History Cards ── */
    .history-item {
        background: #1b2838;
        border-left: 3px solid #d4a843;
        border-radius: 0 8px 8px 0;
        padding: 8px 12px;
        margin: 6px 0;
        font-family: 'Lora', Georgia, serif;
    }
    .history-title {
        color: #e6b800;
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 0.9em;
        font-weight: 500;
        margin: 0;
    }
    .history-author {
        color: #8a9ab5;
        font-size: 0.8em;
        margin: 2px 0 0 0;
    }

    /* ── Buttons (all) ── */
    .stButton > button {
        background: #0d1b2a !important;
        color: #e8dcc8 !important;
        border: 1px solid #3b4f6b !important;
        border-radius: 8px !important;
        font-family: 'Lora', Georgia, serif !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        border-color: #d4a843 !important;
        color: #d4a843 !important;
        background: #1b2838 !important;
    }

    /* ── Footer ── */
    .starry-footer {
        text-align: center;
        color: #4e6382;
        font-family: 'Playfair Display', Georgia, serif;
        font-style: italic;
        font-size: 0.9em;
        padding: 30px 0 15px 0;
        letter-spacing: 0.8px;
        border-top: 1px solid #1b2838;
        margin-top: 40px;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid #1b2838;
        margin: 15px 0;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a1628; }
    ::-webkit-scrollbar-thumb { background: #2c5f8a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a7ca5; }
</style>
"""

st.markdown(STARRY_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    """Load the sentence transformer model (cached)."""
    return SentenceTransformer('BAAI/bge-small-en-v1.5')

@st.cache_resource
def load_data():
    """Load the book data and embeddings (cached)."""
    csv_path = os.path.join(BASE_DIR, 'data', 'enriched_books_from_dump.csv')
    emb_path = os.path.join(BASE_DIR, 'embeddings', 'book_embeddings.npy')
    df = pd.read_csv(csv_path)
    embeddings = np.load(emb_path)

    # Compute weighted_rating (not in the CSV)
    df['average_rating'] = df['average_rating'].fillna(0)
    df['ratings_count'] = df['ratings_count'].fillna(0)
    df['thumbnail'] = df['thumbnail'].fillna('')
    C = df['average_rating'].mean()
    m = df['ratings_count'].quantile(0.90)
    df['weighted_rating'] = df.apply(
        lambda row: (row['ratings_count'] / (row['ratings_count'] + m)) * row['average_rating']
                   + (m / (row['ratings_count'] + m)) * C
        if row['ratings_count'] > 0 else 0,
        axis=1
    )
    return df, embeddings

model = load_model()
df, embeddings = load_data()


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_top_matches(query_vector, doc_vectors, top_n=5):
    """Find top N matching books by cosine similarity."""
    scores = cosine_similarity(query_vector, doc_vectors)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    return top_indices, scores[top_indices]


def build_book_card_html(book):
    """Generate HTML for a styled book card with cover image."""
    title = book.get('title', 'Unknown Title')
    author = book.get('authors', 'Unknown Author')
    rating = book.get('average_rating', 0)
    category = book.get('categories', 'Uncategorized')
    year = book.get('published_year', 'N/A')
    thumbnail = book.get('thumbnail', '')
    score = book.get('match_score', None)

    # Year formatting
    try:
        year = str(int(float(year)))
    except (ValueError, TypeError):
        year = 'N/A'

    # Star rating
    full_stars = int(rating)
    half_star = '½' if (rating - full_stars) >= 0.25 else ''
    stars = '⭐' * full_stars + half_star

    # Cover image or placeholder
    if thumbnail and str(thumbnail).startswith('http'):
        cover_html = f'<img class="book-cover" src="{thumbnail}" alt="{title}">'
    else:
        cover_html = '<div class="book-cover-placeholder">📖</div>'

    # Match score line
    score_html = f'<p class="book-meta">Match: {score:.1%}</p>' if score else ''

    return f"""
    <div class="book-card">
        {cover_html}
        <div class="book-info">
            <p class="book-title">{title}</p>
            <p class="book-author">by {author}</p>
            <p class="book-rating">{stars} {rating:.1f}/5</p>
            <p class="book-meta">{category} &middot; {year}</p>
            {score_html}
        </div>
    </div>
    """


def render_book_cards(books):
    """Render book cards as separate st.markdown calls (ensures HTML renders)."""
    st.markdown('<p class="section-header">📖 Recommended Books</p>', unsafe_allow_html=True)
    for book in books:
        st.markdown(build_book_card_html(book), unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────

# Messages store: [{"role": "user"/"assistant", "content": "text", "books": [list of dicts]}]
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rec_history' not in st.session_state:
    st.session_state.rec_history = []


# ─────────────────────────────────────────────
#  SIDEBAR — Recommendation History
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="section-header">📚 Recommendation History</p>', unsafe_allow_html=True)

    if st.session_state.rec_history:
        for item in reversed(st.session_state.rec_history[-20:]):
            st.markdown(f"""
                <div class="history-item">
                    <p class="history-title">{item['title']}</p>
                    <p class="history-author">by {item['author']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="color: #4e6382; font-style: italic; font-family: Lora, serif;">'
            'No recommendations yet. Ask me anything!</p>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.rec_history = []
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────

# Header
st.markdown('<h1 class="main-title">📚 The AI Librarian</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Your personal guide to the world of books</p>', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📚"):
        st.markdown(msg["content"])

    # Render book cards OUTSIDE the chat bubble (so HTML renders properly)
    if msg["role"] == "assistant" and msg.get("books"):
        render_book_cards(msg["books"])

# Chat input
if prompt := st.chat_input("Ask me for a book recommendation..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt, "books": []})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Process with RAG
    with st.spinner("✨ Searching the library..."):

        # 1. Encode query & find matches
        query_vector = model.encode([prompt])
        top_indices, top_scores = get_top_matches(query_vector, embeddings, top_n=5)
        matched_books = df.iloc[top_indices].copy()
        matched_books['match_score'] = top_scores

        # 2. Build context for LLM
        context_parts = []
        for _, row in matched_books.iterrows():
            ctx = str(row.get('ai_context', row.get('description', '')))
            if ctx and ctx.lower() != 'nan':
                context_parts.append(f"Title: {row['title']} by {row['authors']}\n{ctx[:300]}")
        context_text = "\n\n---\n\n".join(context_parts)

        # 3. Get LLM response
        try:
            chat_history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]
            llm_response = ask_librarian(prompt, context_text, chat_history)
        except Exception as e:
            llm_response = f"I found some great books for you! *(LLM offline — {str(e)[:80]})*"

        # 4. Build book data list (plain dicts, not HTML)
        book_list = []
        for _, row in matched_books.iterrows():
            book_list.append({
                'title': row.get('title', 'Unknown'),
                'authors': row.get('authors', 'Unknown'),
                'average_rating': row.get('average_rating', 0),
                'categories': row.get('categories', 'Uncategorized'),
                'published_year': row.get('published_year', 'N/A'),
                'thumbnail': row.get('thumbnail', ''),
                'match_score': row.get('match_score', 0),
            })

        # 5. Display LLM response in chat bubble
        with st.chat_message("assistant", avatar="📚"):
            st.markdown(llm_response)

        # 6. Render book cards OUTSIDE the chat bubble
        render_book_cards(book_list)

        # 7. Update recommendation history (deduplicate)
        for book in book_list:
            if not any(h['title'] == book['title'] for h in st.session_state.rec_history):
                st.session_state.rec_history.append({
                    'title': book['title'],
                    'author': book['authors'],
                })

        # 8. Save to session state (text only — NO HTML)
        st.session_state.messages.append({
            "role": "assistant",
            "content": llm_response,
            "books": book_list,
        })

    st.rerun()

# Footer
st.markdown(
    '<div class="starry-footer">"I often think that the night is more alive '
    'and more richly colored than the day." — Vincent van Gogh</div>',
    unsafe_allow_html=True
)
