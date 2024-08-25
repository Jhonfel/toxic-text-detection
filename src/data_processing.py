import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any

# Configuración inicial
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('spanish')).union({'q', 'si', 'ser', 'va'})
nlp = spacy.load("es_core_news_sm", disable=["ner", "parser", "tagger"])

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV.

    Args:
        file_path (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    return pd.read_csv(file_path)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame eliminando filas con '#ERROR!', mensajes vacíos y etiquetas inválidas.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame limpio.
    """
    df = df[~df['message'].str.contains('#ERROR!', case=False, na=False)]
    df = df[df['message'].str.strip() != '']
    df = df[df['label'].isin([0, 1])]
    df = df[df['message'].apply(lambda x: bool(re.search(r'[a-zA-Z0-9]', x)))]
    df.drop_duplicates(subset=['message', 'label'], keep='first', inplace=True)
    return df.reset_index(drop=True)

def preprocess_text(text: str) -> str:
    """
    Preprocesa el texto: convierte a minúsculas, elimina URLs, reemplaza menciones y números.

    Args:
        text (str): Texto a preprocesar.

    Returns:
        str: Texto preprocesado.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '[MENCION]', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\b\d+\b', '[NUMERO]', text)
    text = re.sub(r'[^\w\s\[MENCION\]\[NUMERO\]#]|_', '', text)
    words = [word for word in text.split() if word not in STOP_WORDS]
    return ' '.join(words).strip()

def lemmatize_text(text: str) -> str:
    """
    Lematiza un texto preservando tokens especiales.

    Args:
        text (str): Texto a lematizar.

    Returns:
        str: Texto lematizado.
    """
    tokens = re.findall(r'\[MENCION\]|\[NUMERO\]|\S+', text)
    lemmatized_tokens = []
    for token in tokens:
        if token in ['[MENCION]', '[NUMERO]']:
            lemmatized_tokens.append(token)
        else:
            doc = nlp(token)
            lemmatized_tokens.append(doc[0].lemma_)
    return " ".join(lemmatized_tokens)

def extract_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae características básicas del texto.

    Args:
        df (pd.DataFrame): DataFrame con columna 'message'.

    Returns:
        pd.DataFrame: DataFrame con características básicas añadidas.
    """
    df['processed_length'] = df['processed_message'].str.len()
    df['processed_word_count'] = df['processed_message'].str.split().str.len()
    df['processed_avg_word_length'] = df['processed_message'].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
    )
    df['uppercase_ratio'] = df['message'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    df['mencion_count'] = df['processed_message'].str.count('\[MENCION\]')
    df['numero_count'] = df['processed_message'].str.count('\[NUMERO\]')
    return df

def extract_tfidf_features(df: pd.DataFrame, max_features: int = 1000) -> pd.DataFrame:
    """
    Extrae características TF-IDF del texto lematizado.

    Args:
        df (pd.DataFrame): DataFrame con columna 'lemmatized_message'.
        max_features (int): Número máximo de características TF-IDF.

    Returns:
        pd.DataFrame: DataFrame con características TF-IDF añadidas.
    """
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['lemmatized_message'])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out()
    )
    return pd.concat([df, tfidf_df], axis=1)

def process_data(input_path: str, output_path: str) -> None:
    """
    Procesa los datos desde el archivo de entrada y guarda el resultado.

    Args:
        input_path (str): Ruta al archivo de entrada.
        output_path (str): Ruta para guardar el archivo procesado.
    """
    df = load_data(input_path)
    df = clean_dataframe(df)
    df['processed_message'] = df['message'].apply(preprocess_text)
    df['lemmatized_message'] = df['processed_message'].apply(lemmatize_text)
    df = extract_basic_features(df)
    df = extract_tfidf_features(df)
    df.to_csv(output_path, index=False)
    print(f"Datos procesados guardados en: {output_path}")

if __name__ == "__main__":
    input_file = '../data/raw/data_toxic.csv'
    output_file = '../data/processed/data_toxic_features.csv'
    process_data(input_file, output_file)