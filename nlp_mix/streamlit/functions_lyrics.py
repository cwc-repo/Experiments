import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from openai import OpenAI
import json
from typing import Optional

# Configura tu clave API
credentials_path = ""
with open(credentials_path, 'r') as file:
    credentials = json.load(file)

api_keyLlyrics = credentials["chat_openai"]
client = OpenAI(api_key=api_keyLlyrics)

def analyze_song_lyrics_semantically(
    lyrics_1: str,
    artist_1: str,
    lyrics_2: str,
    artist_2: str,
    model: str = "gpt-4"
) -> Optional[str]:
    prompt = f"""
Quiero que analices semánticamente dos fragmentos de letras de canciones escritas en español. Identifica qué emociones, ideas o experiencias tienen en común, incluso si las expresan de forma diferente. Si es posible, encuentra paralelismos temáticos, sentimentales o narrativos.

Letra 1:
🎵 {lyrics_1.strip() } 🎵 del artista {artist_1}

Letra 2:
🎵 {lyrics_2.strip()} 🎵 del artista {artist_2}

Ahora, escribe un análisis detallado y comparativo de ambas letras, destacando similitudes emocionales o conceptuales relevantes.
"""
    response = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": "Eres un experto en análisis de letras de canciones y emociones humanas."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=400
    )
    
    return response.choices[0].message.content if response.choices else None


def jaccard_distance(list1, list2):
    return 1 - jaccard_similarity(list1, list2)

def comb_from_artis(dict_df, artist):
    artist_combs = []
    for comb in dict_df.keys():
        if artist in dict_df[comb]['artist'].unique():
            artist_combs.append(comb)
    return artist_combs
def title_from_comb_artist(dict_df, artist, comb):
    df = dict_df[comb]
    df_artist = df[df['artist']==artist]

    if len(df_artist)==0:
        print('Not Artist for comb')
    else:
        return df_artist['title'].unique()
def verse_from_attributes(dict_df,artist,comb,title):
    df = dict_df[comb]
    df_filter = df[(df['artist']==artist)&(df['title']==title)]
    if len(df_filter)==0:
        print('Not verse for comb')
    else:
        return df_filter['verse_group'].unique()
def random_verse_from_attributes(dict_df, artist, comb, title):
    df = dict_df[comb]
    df_filter = df[(df['artist'] == artist) & (df['title'] == title)]
    
    if df_filter.empty:
        print('No verses found for this combination.')
        return None
    else:
        verses = df_filter['verse_group'].dropna().unique()
        if len(verses) == 0:
            print('No verses available.')
            return None
        return random.choice(verses)
def att_from_verse(dict_df,verse):
    att = {}
    for comb in dict_df.keys():
        df = dict_df[comb]
        if verse in df['verse_group'].unique():
            df_verse = df[df['verse_group']==verse].reset_index()
            att = dict(df_verse.iloc[0])
    return att  
def cluster_from_verse(dict_df,verse):
    df_cluster = pd.DataFrame()
    for comb in dict_df.keys():
        df = dict_df[comb]
        if verse in df['verse_group'].unique():
            df_verse = df[df['verse_group']==verse].reset_index()
            cluster_verse = df_verse.iloc[0]['cluster_label']
            artist = df_verse.iloc[0]['artist']
            df_cluster = df[(df['cluster_label']==cluster_verse)&
                            (df['verse_group']!=verse)
                            #(df['artist']!=artist)
                            ]
    return df_cluster.drop_duplicates(subset=['verse_group'])
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0
def top_k_jaccard(df, keywords_column, query_keywords, k=5):
    query_set = set(query_keywords)

    # Calcular distancia de Jaccard para cada fila
    distances = []
    for keywords in df[keywords_column]:
        row_set = set(keywords)
        distance = jaccard_distance(query_set, row_set)
        distances.append(distance)

    df_copy = df.copy()
    df_copy["jaccard_distance"] = distances

    # Ordenar por menor distancia y retornar top-k
    return df_copy.sort_values("jaccard_distance").head(k).reset_index(drop=True)
def top_k_cosine_from_keywords(df, keywords_column, query_keywords, k=5):
    # Convertir lista de keywords a texto (espacios entre palabras)
    df_keywords_text = df[keywords_column].apply(lambda x: ' '.join(x))
    query_text = ' '.join(query_keywords)

    # Crear matriz TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_keywords_text.tolist() + [query_text])

    # Última fila es la query
    query_vector = tfidf_matrix[-1]
    data_matrix = tfidf_matrix[:-1]

    # Calcular distancias de coseno
    distances = cosine_distances(data_matrix, query_vector).flatten()

    # Crear nuevo DataFrame con distancias
    df_copy = df.copy()
    df_copy["cosine_distance"] = distances

    # Ordenar por menor distancia
    return df_copy.sort_values("cosine_distance").head(k).reset_index(drop=True)