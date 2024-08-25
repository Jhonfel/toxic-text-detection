import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from typing import List, Tuple
import os

# Asegúrate de que NLTK tenga los recursos necesarios
nltk.download('punkt')

def load_processed_data(file_path: str) -> pd.DataFrame:
    """
    Carga los datos procesados desde un archivo CSV.
    
    Args:
        file_path (str): Ruta al archivo CSV con los datos procesados.
    
    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    return pd.read_csv(file_path)

def create_word_embeddings(df: pd.DataFrame, vector_size: int = 100, window: int = 5, min_count: int = 1) -> Word2Vec:
    """
    Crea embeddings de palabras utilizando Word2Vec.
    
    Args:
        df (pd.DataFrame): DataFrame con la columna 'lemmatized_message'.
        vector_size (int): Dimensionalidad de los vectores de palabras.
        window (int): Tamaño máximo de la distancia entre la palabra actual y la predicha.
        min_count (int): Ignora todas las palabras con una frecuencia total menor a esta.
    
    Returns:
        Word2Vec: Modelo de Word2Vec entrenado.
    """
    sentences = [word_tokenize(text) for text in df['lemmatized_message']]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

def get_document_vector(text: str, model: Word2Vec) -> np.ndarray:
    """
    Obtiene el vector de documento promediando los vectores de palabras.
    
    Args:
        text (str): Texto del documento.
        model (Word2Vec): Modelo de Word2Vec entrenado.
    
    Returns:
        np.ndarray: Vector del documento.
    """
    words = word_tokenize(text)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def create_document_embeddings(df: pd.DataFrame, model: Word2Vec) -> pd.DataFrame:
    """
    Crea embeddings de documentos para cada mensaje en el DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame con la columna 'lemmatized_message'.
        model (Word2Vec): Modelo de Word2Vec entrenado.
    
    Returns:
        pd.DataFrame: DataFrame con los embeddings de documentos añadidos.
    """
    document_vectors = df['lemmatized_message'].apply(lambda x: get_document_vector(x, model))
    doc_embedding_df = pd.DataFrame(document_vectors.tolist(), columns=[f'embed_{i}' for i in range(model.vector_size)])
    return pd.concat([df, doc_embedding_df], axis=1)

def select_best_features(X: pd.DataFrame, y: pd.Series, k: int = 100) -> List[str]:
    """
    Selecciona las mejores características basadas en la prueba de chi-cuadrado.
    
    Args:
        X (pd.DataFrame): DataFrame con las características.
        y (pd.Series): Serie con las etiquetas.
        k (int): Número de mejores características a seleccionar.
    
    Returns:
        List[str]: Lista de nombres de las mejores características.
    """
    selector = SelectKBest(chi2, k=k)
    selector.fit(X, y)
    feature_names = X.columns[selector.get_support()].tolist()
    return feature_names

def visualize_feature_importance(feature_names: List[str], importances: np.ndarray, title: str):
    """
    Visualiza la importancia de las características.
    
    Args:
        feature_names (List[str]): Nombres de las características.
        importances (np.ndarray): Importancias de las características.
        title (str): Título del gráfico.
    """
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances, y=feature_names)
    plt.title(title)
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.show()

def analyze_correlations(df: pd.DataFrame, features: List[str]):
    """
    Analiza y visualiza las correlaciones entre características.
    
    Args:
        df (pd.DataFrame): DataFrame con las características.
        features (List[str]): Lista de nombres de características a analizar.
    """
    corr_matrix = df[features + ['label']].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación de Características')
    plt.tight_layout()
    plt.show()

def engineer_features(input_path: str, output_path: str):
    """
    Realiza la ingeniería de características y guarda los resultados.
    
    Args:
        input_path (str): Ruta al archivo de entrada con datos procesados.
        output_path (str): Ruta para guardar el archivo con características ingenierizadas.
    """
    df = load_processed_data(input_path)
    
    # Crear embeddings de palabras y documentos
    word2vec_model = create_word_embeddings(df)
    df = create_document_embeddings(df, word2vec_model)
    
    # Seleccionar mejores características
    feature_columns = [col for col in df.columns if col not in ['message', 'processed_message', 'lemmatized_message', 'label']]
    best_features = select_best_features(df[feature_columns], df['label'])
    
    # Visualizar importancia de características
    visualize_feature_importance(best_features[:20], df[best_features].corr()['label'].abs().sort_values(ascending=False)[:20], 'Top 20 Características más Importantes')
    
    # Analizar correlaciones
    analyze_correlations(df, best_features[:20])
    
    # Guardar DataFrame con características seleccionadas
    output_df = df[['message', 'label'] + best_features]
    output_df.to_csv(output_path, index=False)
    print(f"Características ingenierizadas guardadas en: {output_path}")

if __name__ == "__main__":
    input_file = '../data/processed/data_toxic_processed.csv'
    output_file = '../data/processed/data_toxic_engineered_features.csv'
    engineer_features(input_file, output_file)