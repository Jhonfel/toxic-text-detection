# src/__init__.py

from .data_processing import (
    load_data,
    clean_dataframe,
    preprocess_text,
    lemmatize_text,
    extract_basic_features,
    extract_tfidf_features,
    process_data
)

from .feature_engineering import (
    load_processed_data,
    create_word_embeddings,
    create_document_embeddings,
    select_best_features,
    visualize_feature_importance,
    analyze_correlations,
    engineer_features
)

__all__ = [
    # Funciones de data_processing
    "load_data",
    "clean_dataframe",
    "preprocess_text",
    "lemmatize_text",
    "extract_basic_features",
    "extract_tfidf_features",
    "process_data",
    
    # Funciones de feature_engineering
    "load_processed_data",
    "create_word_embeddings",
    "create_document_embeddings",
    "select_best_features",
    "visualize_feature_importance",
    "analyze_correlations",
    "engineer_features"
]

# Puedes agregar información sobre la versión del paquete
__version__ = "0.1.0"