import pickle
import time
import pandas as pd
import streamlit as st
from functions_lyrics import (att_from_verse,cluster_from_verse, top_k_jaccard, 
                              top_k_cosine_from_keywords, analyze_song_lyrics_semantically)

def show_animated_text(texto, velocidad=0.05):
    espacio = st.empty()
    resultado = ""
    for char in texto:
        resultado += char
        espacio.markdown(f"**{resultado}**")
        time.sleep(velocidad)


dict_sen = {'Negativo': 'NEG', 'Neutro':'NEU', 'Positivo':'POS'}
dict_emo = {'enojo':'anger','disgustado':'disgust'}

with open('clusters_dataframes.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# Extraer sentimientos únicos
sen = sorted(set(dict_sen.keys()))

# Interfaz
st.title("🎭 Filtro por Sentimiento, Emoción y Artista")

# Selección de sentimiento
sent = st.selectbox("Selecciona un sentimiento", sen)
sen_format = dict_sen[sent]

# Emociones disponibles para ese sentimiento
emociones_filtradas = sorted([e for (s, e) in data_dict.keys() if s == sen_format])

# Selección de emoción
emo = st.selectbox("Selecciona una emoción", emociones_filtradas)

# Artistas disponibles para esa combinación
df_fix = data_dict[(sen_format, emo)]
artistas = df_fix['artist'].unique()

# Mostrar artistas si existen
if len(artistas) > 0:
    artist = st.selectbox("Selecciona un artista", artistas)
    df_artist = df_fix[df_fix['artist'] == artist]

    # Títulos disponibles para ese artista
    songs = df_artist['title'].unique()
    title = st.selectbox("Selecciona un título", songs)

    # Versos disponibles
    verse_list = df_artist[df_artist['title'] == title]['verse_group'].unique()

    if len(verse_list) > 0:
        #random_verse = random.choice(verse_list)
        #st.markdown(f"🎵 Verso aleatorio de **{title}** por **{artist}**: \n\n> *{random_verse}*")
        verse = st.selectbox("Selecciona un verso", verse_list)

        # Botón de similitud semántica con ícono
        if st.button("💿🎧 Calcula similitud semántica"):
            # Aquí va tu cálculo de similitud (simulado con texto por ahora)
            st.success("🔍 Analizando similitud semántica...")
            #verse_proof = random_verse_from_attributes(data_dict,artist,(sen_format, emo),title)
            verse_proof = verse
            data_verse = att_from_verse(data_dict,verse_proof)
            df_cluster_verse = cluster_from_verse(data_dict,verse_proof)
            df_top_jacc = top_k_jaccard(df_cluster_verse,"key_words", data_verse['key_words'], k=5)
            df_top_emb = top_k_cosine_from_keywords(df_cluster_verse,"key_words", data_verse['key_words'], k=5)
            i=0
            simil_artist = df_top_emb.iloc[i]['artist']
            simil_song = df_top_emb.iloc[i]['title']
            simil_verse = df_top_emb.iloc[i]['verse_group']
            try:
                if verse_proof==simil_verse:
                        i =1
                        simil_artist = df_top_emb.iloc[i]['artist']
                        simil_song = df_top_emb.iloc[i]['title']
                        simil_verse = df_top_emb.iloc[i]['verse_group']
                text = f"Si **{artist}** en **{title}** dijo: 🎵 ...{verse}...🎵"
                simil_text = f"Entonces **{simil_artist}** en **{simil_song}** dice: 🎵 ...{simil_verse}...🎵"
                st.markdown(f"🧠 {text}")
                show_animated_text(simil_text)

                text_interpretation = analyze_song_lyrics_semantically(verse,artist,simil_verse,simil_artist)
                st.markdown(f"🤖 Interpretacion por LLM")
                show_animated_text(text_interpretation)
            except:
                simil_text = f"Así dijo **{simil_artist}** en **{simil_song}**: 🎵 ...{simil_verse}...🎵"
                #st.markdown(f"🧠 Resultado: {simil_text}")
                show_animated_text(simil_text)



    else:
        st.warning("No hay versos disponibles para esta canción.")
else:
    st.warning("No hay artistas ni títulos para esta combinación.")
