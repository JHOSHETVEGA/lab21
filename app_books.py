import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Recomendación de Libros",
    page_icon="📚",
    layout="wide"
)

# Conexión a MongoDB
@st.cache_resource
def get_database_connection():
    try:
        client = MongoClient("mongodb+srv://jvegafcalma_db_user:72809780@cluster0.wvvk0ov.mongodb.net/")
        db = client["book_recommendation_db"]
        return db, True
    except Exception as e:
        st.error(f"Error al conectar con MongoDB: {e}")
        return None, False

# Inicialización de la base de datos
def initialize_database(db):
    if db.books.count_documents({}) == 0:
        books_data = [
            {
                "title": "Cien años de soledad",
                "author": "Gabriel García Márquez",
                "genre": "Realismo mágico",
                "description": "La historia épica de la familia Buendía en el mítico pueblo de Macondo.",
                "rating": 9.5
            },
            {
                "title": "1984",
                "author": "George Orwell",
                "genre": "Distopía, Política, Ciencia Ficción",
                "description": "Una sociedad totalitaria donde el Gran Hermano vigila cada movimiento.",
                "rating": 9.2
            },
            {
                "title": "Orgullo y Prejuicio",
                "author": "Jane Austen",
                "genre": "Romance, Clásico",
                "description": "La historia de amor y orgullo entre Elizabeth Bennet y el Sr. Darcy.",
                "rating": 8.9
            },
            {
                "title": "El nombre del viento",
                "author": "Patrick Rothfuss",
                "genre": "Fantasía, Aventura",
                "description": "La vida de Kvothe, un joven prodigio con talento para la magia y la música.",
                "rating": 9.0
            },
            {
                "title": "Los pilares de la Tierra",
                "author": "Ken Follett",
                "genre": "Histórico, Drama",
                "description": "La construcción de una catedral en la Inglaterra medieval llena de intrigas.",
                "rating": 8.8
            },
            {
                "title": "El Hobbit",
                "author": "J.R.R. Tolkien",
                "genre": "Fantasía, Aventura",
                "description": "Bilbo Bolsón emprende un viaje inesperado para recuperar un tesoro custodiado por un dragón.",
                "rating": 9.1
            },
            {
                "title": "Crónica de una muerte anunciada",
                "author": "Gabriel García Márquez",
                "genre": "Misterio, Realismo mágico",
                "description": "Una historia donde todos saben que va a ocurrir un asesinato, menos la víctima.",
                "rating": 8.6
            },
            {
                "title": "El alquimista",
                "author": "Paulo Coelho",
                "genre": "Ficción, Filosofía",
                "description": "Un joven pastor andaluz sigue su sueño en busca de un tesoro y de sí mismo.",
                "rating": 8.4
            },
            {
                "title": "Harry Potter y la piedra filosofal",
                "author": "J.K. Rowling",
                "genre": "Fantasía, Juvenil, Aventura",
                "description": "Un niño descubre que es mago y asiste a una escuela mágica llamada Hogwarts.",
                "rating": 9.3
            },
            {
                "title": "Don Quijote de la Mancha",
                "author": "Miguel de Cervantes",
                "genre": "Clásico, Aventura, Humor",
                "description": "Un hidalgo pierde la cordura y sale a recorrer España como caballero andante.",
                "rating": 9.0
            }
        ]

        db.books.insert_many(books_data)
        return True
    return False

# Obtener todos los libros
def get_all_books(db):
    return list(db.books.find({}, {"_id": 0}))

# Generar recomendaciones por contenido
def get_content_recommendations(db, selected_title, n_recommendations=5):
    df = pd.DataFrame(get_all_books(db))
    if df.empty or selected_title not in df['title'].values:
        return []

    genre_col = "genre" if "genre" in df.columns else "genres"
    df["description"] = df["description"].fillna("")
    df[genre_col] = df[genre_col].fillna("")
    df["author"] = df["author"].fillna("")

    df["text"] = df["description"].astype(str) + " " + df[genre_col].astype(str) + " " + df["author"].astype(str)

    # ✅ Lista de stopwords en español
    spanish_stopwords = [
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
        "las", "por", "un", "para", "con", "no", "una", "su", "al",
        "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este",
        "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre"
    ]

    vectorizer = TfidfVectorizer(stop_words=spanish_stopwords)
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = df.index[df["title"] == selected_title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]

    recommendations = df.iloc[[i[0] for i in sim_scores]].to_dict(orient="records")
    return recommendations

# Mostrar tarjetas de libros
def display_book_cards(books):
    cols = st.columns(3)
    for i, book in enumerate(books):
        # Evita KeyError usando .get()
        title = book.get("title", "Título desconocido")
        author = book.get("author", "Autor desconocido")
        genre = book.get("genre") or book.get("genres", "Género no especificado")
        rating = book.get("rating", "N/A")
        description = book.get("description", "Sin descripción disponible")

        with cols[i % 3]:
            st.markdown(f"""
            <div style="border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px">
                <h3>{title}</h3>
                <p><strong>Autor:</strong> {author}</p>
                <p><strong>Género:</strong> {genre}</p>
                <p><strong>Calificación:</strong> {rating}/10</p>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

# Interfaz principal
def main():
    st.title("📚 Sistema de Recomendación de Libros")
    db, flag = get_database_connection()
    if not flag:
        st.error("Error al conectar con la base de datos.")
        return

    if initialize_database(db):
        st.success("Base de datos inicializada con datos de ejemplo.")

    books = get_all_books(db)
    titles = [b["title"] for b in books]

    selected_book = st.selectbox("Selecciona un libro para obtener recomendaciones:", titles)

    if selected_book:
        st.subheader(f"📖 Libro seleccionado: {selected_book}")
        recs = get_content_recommendations(db, selected_book)

        if recs:
            st.subheader("🔍 Recomendaciones similares:")
            display_book_cards(recs)
        else:
            st.warning("No se encontraron recomendaciones para este libro.")

    st.divider()
    st.subheader("📘 Catálogo completo")
    display_book_cards(books)

if __name__ == "__main__":
    main()
