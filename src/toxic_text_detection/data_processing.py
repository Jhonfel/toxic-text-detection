import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import unicodedata
from tqdm import tqdm

# Descargar recursos necesarios
nltk.download('stopwords')
nltk.download('punkt')
tqdm.pandas()

# Cargar modelo de spaCy
nlp = spacy.load("es_core_news_sm", disable=["ner", "parser", "tagger"])

# Configuración de stopwords
STOP_WORDS = set(stopwords.words('spanish'))
STOP_WORDS.update(['q', 'si', 'ser', 'va'])

def load_data(filepath):
    """
    Carga los datos desde un archivo CSV.
    """
    return pd.read_csv(filepath)

def preprocess_text(text):
    """
    Preprocesa el texto: convierte a minúsculas, elimina URLs, elimina caracteres especiales y stopwords,
    reemplaza las menciones (@username) con un token estándar, mantiene los hashtags,
    elimina menciones duplicadas y reemplaza números completos con [NUMERO].
    """
    if pd.isna(text):
        return ""
    
    # Convierte a minúsculas
    text = text.lower()

    # Elimina URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Reemplaza las menciones (@username) con [MENCION]
    text = re.sub(r'@\w+', '[MENCION]', text)

    # Elimina el símbolo '#' pero mantiene el texto del hashtag
    text = re.sub(r'#(\w+)', r'\1', text)

    # Reemplaza números completos con [NUMERO]
    text = re.sub(r'\b\d+\b', '[NUMERO]', text)

    # Elimina caracteres especiales, incluyendo el guion bajo, pero manteniendo [MENCION] y [NUMERO]
    text = re.sub(r'[^\w\s\[MENCION\]\[NUMERO\]#]|_', '', text)

    # Divide el texto en palabras y elimina las stopwords
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]

    # Elimina menciones duplicadas
    mention_found = False
    filtered_words = []
    for word in words:
        if word == '[MENCION]':
            if not mention_found:
                filtered_words.append(word)
                mention_found = True
        else:
            filtered_words.append(word)

    return ' '.join(filtered_words).strip()

def lemmatize_text(text):
    """
    Lematiza un texto individual, preservando tokens especiales como '[MENCION]' y '[NUMERO]',
    incluso cuando están pegados a otras palabras.
    """
    if pd.isna(text):
        return ""

    special_tokens = ['[MENCION]', '[NUMERO]']

    for token in special_tokens:
        text = re.sub(rf'({re.escape(token)})(\S)', r'\1 \2', text)
        text = re.sub(rf'(\S)({re.escape(token)})', r'\1 \2', text)

    tokens = re.findall(r'\[MENCION\]|\[NUMERO\]|\S+', text)

    lemmatized_tokens = []
    for token in tokens:
        if token in special_tokens:
            lemmatized_tokens.append(token)
        else:
            doc = nlp(token)
            lemmatized_tokens.append(doc[0].lemma_)

    return " ".join(lemmatized_tokens)

def remove_accents(text):
    """
    Elimina los acentos de un texto.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def count_offensive_words(text, offensive_words):
    """
    Cuenta el número de palabras ofensivas en el texto.
    """
    text = text.lower()
    return sum(1 for palabra in offensive_words if re.search(r'\b' + re.escape(palabra) + r'\b', text))

def process_data(df, offensive_words):
    """
    Procesa el DataFrame aplicando todas las funciones de preprocesamiento.
    """
    print("Preprocesando texto...")
    df['processed_message'] = df['message'].progress_apply(preprocess_text)
    
    print("Lematizando texto...")
    df['lemmatized_message'] = df['processed_message'].progress_apply(lemmatize_text)
    
    print("Eliminando acentos...")
    df['lemmatized_message'] = df['lemmatized_message'].progress_apply(remove_accents)
    
    print("Contando palabras ofensivas...")
    df['ofensivas_count'] = df['lemmatized_message'].progress_apply(lambda x: count_offensive_words(x, offensive_words))
    
    return df

def save_processed_data(df, output_path):
    """
    Guarda el DataFrame procesado en un archivo CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"DataFrame procesado guardado en: {output_path}")

def process_text_for_inference(text, offensive_words):
    """
    Procesa un texto individual para inferencia, aplicando todas las etapas de preprocesamiento.
    """
    processed_text = preprocess_text(text)
    lemmatized_text = lemmatize_text(processed_text)
    text_without_accents = remove_accents(lemmatized_text)
    offensive_count = count_offensive_words(text_without_accents, offensive_words)
    
    return {
        'processed_message': processed_text,
        'lemmatized_message': text_without_accents,
        'ofensivas_count': offensive_count
    }

