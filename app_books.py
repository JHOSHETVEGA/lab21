# ======================================================
# 📚 SISTEMA DE RECOMENDACIÓN DE LIBROS
# Autor: Joshua Vega
# Versión: 1.1 (corregida)
# ======================================================

import streamlit as st
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------
# CONFIGURACIÓN INICIAL DE LA APP
# ------------------------------------------------------
st.set_page_config(
    page_title="📚 Sistema de Recomendación de Libros",
    layout="wide"
)

# ------------------------------------------------------
# CONEXIÓN A MONGODB
# ------------------------------------------------------
@st.cache_resource
def get_database_connection():
    try:
        client = MongoClient("mongodb+srv://jvegafcalma_db_user:72809780@cluster0.wvvk0ov.mongodb.net/")
        db = client["book_recommendation_db"]
        return db, True
    except Exception as e:
        st.error(f"❌ Error de conexión con MongoDB: {e}")
        return None, False


# ------------------------------------------------------
# CARGA DE DATOS DE EJEMPLO (SOLO SI ESTÁ VACÍA)
# ------------------------------------------------------
def initialize_database(db):
    if db.books.count_documents({}) == 0:
        books_data = [
            {
                "title": "Cien Años de Soledad",
                "author": "Gabriel García Márquez",
                "genres": ["Realismo Mágico", "Literatura Latinoamericana"],
                "description": "La historia multigeneracional de la familia Buendía en el mítico pueblo de Macondo.",
                "rating": 9.5
            },
            {
                "title": "1984",
                "author": "George Orwell",
                "genres": ["Distopía", "Ciencia Ficción"],
                "description": "Una sociedad totalitaria controlada por el Gran Hermano donde la libertad individual no existe.",
                "rating": 9.3
            },
            {
                "title": "El Principito",
                "author": "Antoine de Saint-Exupéry",
                "genres": ["Infantil", "Filosofía"],
                "description": "Un piloto conoce a un pequeño príncipe de otro planeta que le enseña el valor de la amistad y la inocencia.",
                "rating": 8.9
            },
            {
                "title": "Orgullo y Prejuicio",
                "author": "Jane Austen",
                "genres": ["Romance", "Clásico"],
                "description": "La historia de Elizabeth Bennet y el señor Darcy, marcada por las diferencias sociales y los prejuicios.",
                "rating": 8.7
            },
            {
                "title": "Harry Potter y la Piedra Filosofal",
                "author": "J.K. Rowling",
                "genres": ["Fantasía", "Aventura"],
                "description": "Un joven descubre que es un mago y comienza su educación en Hogwarts, una escuela de magia y hechicería.",
                "rating": 9.0
            },
            {
                "title": "El Hobbit",
                "author": "J.R.R. Tolkien",
                "genres": ["Fantasía", "Aventura"],
                "description": "Bilbo Bolsón emprende un viaje lleno de peligros junto a un grupo de enanos para recuperar un tesoro custodiado por un dragón.",
                "rating": 8.8
            },
            {
                "title": "Los Juegos del Hambre",
                "author": "Suzanne Collins",
                "genres": ["Ciencia Ficción", "Acción"],
                "description": "Katniss Everdeen debe luchar por su vida en una competencia televisada en un futuro distópico.",
                "rating": 8.6
            },
            {
                "title": "Crimen y Castigo",
                "author": "Fiódor Dostoyevski",
                "genres": ["Drama", "Psicológico"],
                "description": "Un joven estudiante comete un asesinato y lucha con la culpa moral y la redención.",
                "rating": 9.1
            }
        ]
        db.books.insert_many(books_data)
        return True
    return False


# ------------------------------------------------------
# FUNCIÓN PRINCIPAL DE RECOMENDACIÓN
# ------------------------------------------------------
def get_content_recommendations(db, selected_title, n=5):
    books = list(db.books.find({}, {"_id": 0}))
    df = pd.DataFrame(books)

    if selected_title not in df["title"].values:
        return []

    # --- Limpieza de datos ---
    df["description"] = df["description"].fillna("")
    df["genres"] = df["genres"].apply(lambda g: g if isinstance(g, list) else [])
    df["text"] = df["description"] + " " + df["genres"].apply(lambda g: " ".join(g))
    df["text"] = df["text"].astype(str)

    # --- Vectorización TF-IDF ---
    vectorizer = TfidfVectorizer(stop_words="spanish")
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    # --- Similaridad coseno ---
    similarity = cosine_similarity(tfidf_matrix)

    # Índice del libro seleccionado
    idx = df.index[df["title"] == selected_title][0]

    # Ordenar por similitud descendente
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]

    recommended_books = df.iloc[[i[0] for i in sim_scores]]
    return recommended_books.to_dict(orient="records")


# ------------------------------------------------------
# VISUALIZACIÓN DE LIBROS EN TARJETAS
# ------------------------------------------------------
def display_book_cards(books):
    st.markdown("""
    <style>
    .book-card {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .book-card:hover {
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    for i, book in enumerate(books):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="book-card">
                <h3>{book['title']}</h3>
                <p><strong>Autor:</strong> {book['author']}</p>
                <p><strong>Géneros:</strong> {', '.join(book['genres'])}</p>
                <p><strong>Calificación:</strong> ⭐ {book['rating']}/10</p>
                <p>{book['description']}</p>
            </div>
            """, unsafe_allow_html=True)


# ------------------------------------------------------
# APP PRINCIPAL
# ------------------------------------------------------
def main():
    st.title("📚 Sistema de Recomendación de Libros")

    db, connected = get_database_connection()
    if not connected:
        st.stop()

    # Evita inicialización repetida
    if "initialized" not in st.session_state:
        if initialize_database(db):
            st.success("✅ Base de datos inicializada con datos de ejemplo.")
        st.session_state["initialized"] = True

    all_books = list(db.books.find({}, {"_id": 0, "title": 1}))

    st.subheader("Selecciona un libro para obtener recomendaciones")
    selected_book = st.selectbox("📖 Elige un libro:", [b["title"] for b in all_books])

    if st.button("🔍 Mostrar Recomendaciones"):
        recs = get_content_recommendations(db, selected_book)
        if recs:
            st.markdown(f"### 📗 Libros similares a **{selected_book}**:")
            display_book_cards(recs)
        else:
            st.warning("No se encontraron recomendaciones similares.")

    st.divider()
    st.subheader("📘 Catálogo Completo de Libros")
    books = list(db.books.find({}, {"_id": 0}))
    display_book_cards(books)


# ------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ------------------------------------------------------
if __name__ == "__main__":
    main()
