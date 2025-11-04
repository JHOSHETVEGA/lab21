
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# CONFIGURACIÓN INICIAL
st.set_page_config(page_title="Web Scraper Pro", page_icon="🌐", layout="wide")
st.title(" Web Scraper Pro con Streamlit")
st.markdown("### Extrae datos, genera reportes visuales y analiza patrones ")


# FUNCIÓN DE SCRAPING
def scrape_quotes(url):
    try:
        res = requests.get(url)
        res.raise_for_status()
    except Exception as e:
        st.error(f" Error al acceder a la URL: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(res.text, 'html.parser')
    quotes = soup.select('.quote')
    data = []

    for q in quotes:
        texto = q.select_one('.text').text.strip('“”')
        autor = q.select_one('.author').text.strip()
        tags = [tag.text for tag in q.select('.tags .tag')]
        data.append({
            'Texto': texto,
            'Autor': autor,
            'Tags': tags,
            'Num Tags': len(tags)
        })

    return pd.DataFrame(data)

# INTERFAZ DE USUARIO

url = st.text_input("Ingresa la URL para analizar:", "http://quotes.toscrape.com/page/1/")

if st.button(" Extraer Datos"):
    df = scrape_quotes(url)

    if df.empty:
        st.warning("No se encontraron datos o hubo un error en la extracción.")
    else:
        st.success(f" Se extrajeron {len(df)} citas correctamente.")
        st.dataframe(df, use_container_width=True)

        # MÉTRICAS RESUMEN
        col1, col2, col3 = st.columns(3)
        col1.metric(" Autores únicos", df['Autor'].nunique())
        col2.metric(" Tags únicos", len(set(tag for tags in df['Tags'] for tag in tags)))
        col3.metric(" Promedio de Tags por Cita", round(df['Num Tags'].mean(), 2))

        # GRÁFICOS
        st.markdown("##  Análisis Visual de los Datos")

        # --- Autores más citados ---
        st.subheader(" Autores más citados")
        autores = df['Autor'].value_counts().head(10)
        fig1, ax1 = plt.subplots()
        autores.plot(kind='bar', ax=ax1, color='#4c72b0')
        ax1.set_xlabel("Autor")
        ax1.set_ylabel("Cantidad de citas")
        ax1.set_title("Top 10 Autores más citados")
        st.pyplot(fig1)

        # ---  Tags más populares ---
        st.subheader(" Tags más populares")
        tags_flat = [tag for tags in df['Tags'] for tag in tags]
        tags_counts = Counter(tags_flat)
        top_tags = dict(sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        fig2, ax2 = plt.subplots()
        ax2.barh(list(top_tags.keys())[::-1], list(top_tags.values())[::-1], color='#55a868')
        ax2.set_xlabel("Frecuencia")
        ax2.set_ylabel("Tag")
        ax2.set_title("Top 10 Tags más usados")
        st.pyplot(fig2)

        # ---  Distribución del número de tags por cita ---
        st.subheader("📈 Distribución de la cantidad de tags por cita")
        fig3, ax3 = plt.subplots()
        sns.histplot(df['Num Tags'], bins=range(0, max(df['Num Tags']) + 2), ax=ax3, color='#c44e52')
        ax3.set_xlabel("Número de Tags")
        ax3.set_ylabel("Cantidad de Citas")
        st.pyplot(fig3)

        # ---  Relación entre autores y cantidad promedio de tags ---
        st.subheader("🔍 Promedio de etiquetas por autor")
        avg_tags = df.groupby('Autor')['Num Tags'].mean().sort_values(ascending=False).head(10)
        fig4, ax4 = plt.subplots()
        avg_tags.plot(kind='barh', ax=ax4, color='#8172b2')
        ax4.set_xlabel("Promedio de Tags")
        ax4.set_ylabel("Autor")
        st.pyplot(fig4)

        # ---  Gráfico circular de proporción de citas por autor ---
        st.subheader(" Proporción de citas por autor (Top 5)")
        top_autores = df['Autor'].value_counts().head(5)
        fig5, ax5 = plt.subplots()
        ax5.pie(top_autores.values, labels=top_autores.index, autopct='%1.1f%%', startangle=140)
        ax5.set_title("Distribución de Citas por Autor")
        st.pyplot(fig5)

        
        # DESCARGA DEL DATASET
   
        st.markdown("##  Exportar datos")
        st.download_button(
            label="Descargar dataset en CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='scraped_data.csv',
            mime='text/csv'
        )

