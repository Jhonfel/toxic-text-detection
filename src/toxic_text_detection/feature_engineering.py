import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import joblib
import os

# Asegurar que se han descargado los recursos necesarios de NLTK
nltk.download('punkt')

def load_processed_data(filepath):
    """
    Carga los datos procesados desde un archivo CSV.
    """
    return pd.read_csv(filepath)

def create_tfidf_features(texts, max_features=1000):
    """
    Crea características TF-IDF a partir de los textos.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_matrix, tfidf_vectorizer

def create_word2vec_features(texts, vector_size=100, window=5, min_count=1, workers=4):
    """
    Crea características Word2Vec a partir de los textos.
    """
    tokenized_texts = [word_tokenize(text.lower()) for text in tqdm(texts, desc="Tokenizando textos")]
    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    
    def get_doc_vector(doc):
        words = [word for word in doc if word in w2v_model.wv]
        if len(words) > 0:
            return np.mean(w2v_model.wv[words], axis=0)
        else:
            return np.zeros(vector_size)
    
    doc_vectors = np.array([get_doc_vector(doc) for doc in tqdm(tokenized_texts, desc="Creando vectores de documentos")])
    return doc_vectors, w2v_model

def select_best_features(X, y, k=100):
    """
    Selecciona las mejores k características basadas en la prueba chi-cuadrado.
    """
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector

def create_features(df):
    """
    Crea todas las características para el modelo.
    """
    print("Creando características TF-IDF...")
    tfidf_matrix, tfidf_vectorizer = create_tfidf_features(df['lemmatized_message'])
    
    print("Creando características Word2Vec...")
    w2v_features, w2v_model = create_word2vec_features(df['lemmatized_message'])
    
    print("Combinando características...")
    features = np.hstack((tfidf_matrix.toarray(), w2v_features, df[['ofensivas_count']].values))
    
    print("Seleccionando las mejores características...")
    best_features, selector = select_best_features(features, df['label'])
    
    return best_features, tfidf_vectorizer, w2v_model, selector

def save_feature_models(tfidf_vectorizer, w2v_model, selector, output_dir):
    """
    Guarda los modelos de características para su uso posterior.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(tfidf_vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))
    joblib.dump(w2v_model, os.path.join(output_dir, 'w2v_model.joblib'))
    joblib.dump(selector, os.path.join(output_dir, 'feature_selector.joblib'))
    print(f"Modelos de características guardados en {output_dir}")

def load_feature_models(input_dir):
    """
    Carga los modelos de características guardados.
    """
    tfidf_vectorizer = joblib.load(os.path.join(input_dir, 'tfidf_vectorizer.joblib'))
    w2v_model = joblib.load(os.path.join(input_dir, 'w2v_model.joblib'))
    selector = joblib.load(os.path.join(input_dir, 'feature_selector.joblib'))
    return tfidf_vectorizer, w2v_model, selector

def process_text_for_model(text, tfidf_vectorizer, w2v_model, selector, offensive_words):
    """
    Procesa un texto individual para la inferencia del modelo.
    """
    from data_processing import process_text_for_inference
    
    # Procesar el texto
    processed = process_text_for_inference(text, offensive_words)
    
    # Crear características TF-IDF
    tfidf_features = tfidf_vectorizer.transform([processed['lemmatized_message']]).toarray()
    
    # Crear características Word2Vec
    tokenized_text = word_tokenize(processed['lemmatized_message'].lower())
    w2v_features = np.mean([w2v_model.wv[word] for word in tokenized_text if word in w2v_model.wv], axis=0)
    if np.isnan(w2v_features).any():
        w2v_features = np.zeros(w2v_model.vector_size)
    
    # Combinar todas las características
    all_features = np.hstack((tfidf_features, w2v_features.reshape(1, -1), np.array([[processed['ofensivas_count']]])))
    
    # Seleccionar las mejores características
    best_features = selector.transform(all_features)
    
    return best_features

