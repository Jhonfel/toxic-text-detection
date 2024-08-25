import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint
import joblib
import os
from tqdm import tqdm

from .data_processing import process_text_for_inference
from .data_processing import process_data

from .feature_engineering import create_tfidf_features, create_word2vec_features, select_best_features

class ToxicDetectionModel:
    def __init__(self, offensive_words=None):
        self.offensive_words = offensive_words or []
        self.tfidf_vectorizer = None
        self.w2v_model = None
        self.selector = None
        self.model = None

    def preprocess_data(self, df):
        """
        Preprocesa los datos utilizando las funciones de data_processing.py
        """
        return process_data(df, self.offensive_words)

    def create_features(self, texts):
        """
        Crea características utilizando TF-IDF y Word2Vec
        """
        print("Creando características TF-IDF...")
        tfidf_matrix, self.tfidf_vectorizer = create_tfidf_features(texts)
        
        print("Creando características Word2Vec...")
        w2v_features, self.w2v_model = create_word2vec_features(texts)
        
        print("Combinando características...")
        features = np.hstack((tfidf_matrix.toarray(), w2v_features))
        
        return features

    def select_features(self, X, y):
        """
        Selecciona las mejores características
        """
        print("Seleccionando las mejores características...")
        # Asegurar que todas las características sean no negativas
        X_non_negative = np.abs(X)
        best_features, self.selector = select_best_features(X_non_negative, y)
        return best_features

    def train(self, X, y, param_dist=None, n_iter=100, cv=5):
        """
        Entrena el modelo XGBoost con búsqueda aleatoria de hiperparámetros
        """
        if param_dist is None:
            param_dist = {
                'n_estimators': randint(100, 1000),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4)
            }

        self.model = XGBClassifier(random_state=42)
        
        random_search = RandomizedSearchCV(
            self.model, param_distributions=param_dist, n_iter=n_iter,
            scoring='f1', n_jobs=-1, cv=cv, verbose=1, random_state=42
        )
        random_search.fit(X, y)
        
        self.model = random_search.best_estimator_
        return random_search.best_params_, random_search.best_score_

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo y retorna el informe de clasificación y la matriz de confusión
        """
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return report, cm

    def plot_confusion_matrix(self, cm):
        """
        Grafica la matriz de confusión
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.show()

    def predict(self, text):
        """
        Predice la toxicidad de un texto dado
        """
        processed = process_text_for_inference(text, self.offensive_words)
        tfidf_features = self.tfidf_vectorizer.transform([processed['lemmatized_message']]).toarray()
        w2v_features = self.w2v_model.wv.get_mean_vector(processed['lemmatized_message'].split())
        
        features = np.hstack((tfidf_features, w2v_features.reshape(1, -1)))
        selected_features = self.selector.transform(features)
        
        prediction = self.model.predict(selected_features)
        probability = self.model.predict_proba(selected_features)[0][1]
        
        return prediction[0], probability

    def save(self, output_dir):
        """
        Guarda el modelo y sus componentes
        """
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(output_dir, 'xgboost_model.joblib'))
        joblib.dump(self.tfidf_vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))
        joblib.dump(self.w2v_model, os.path.join(output_dir, 'w2v_model.joblib'))
        joblib.dump(self.selector, os.path.join(output_dir, 'feature_selector.joblib'))
        joblib.dump(self.offensive_words, os.path.join(output_dir, 'offensive_words.joblib'))
        print(f"Modelo y componentes guardados en {output_dir}")

    @classmethod
    def load(cls, input_dir):
        """
        Carga el modelo y sus componentes
        """
        model = cls()
        model.model = joblib.load(os.path.join(input_dir, 'xgboost_model.joblib'))
        model.tfidf_vectorizer = joblib.load(os.path.join(input_dir, 'tfidf_vectorizer.joblib'))
        model.w2v_model = joblib.load(os.path.join(input_dir, 'w2v_model.joblib'))
        model.selector = joblib.load(os.path.join(input_dir, 'feature_selector.joblib'))
        model.offensive_words = joblib.load(os.path.join(input_dir, 'offensive_words.joblib'))
        return model

