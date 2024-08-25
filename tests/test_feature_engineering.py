import unittest
import pandas as pd
import numpy as np
from toxic_text_detection.feature_engineering import (
    create_tfidf_features,
    create_word2vec_features,
    select_best_features,
    create_features,
    process_text_for_model
)
import os
import tempfile

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.sample_texts = [
            "Este es un texto de prueba",
            "Otro texto para probar características"
        ]
        self.sample_df = pd.DataFrame({
            'lemmatized_message': self.sample_texts,
            'ofensivas_count': [0, 1],
            'label': [0, 1]
        })
        self.temp_dir = tempfile.mkdtemp()

    def test_create_tfidf_features(self):
        tfidf_matrix, tfidf_vectorizer = create_tfidf_features(self.sample_texts)
        self.assertIsNotNone(tfidf_matrix)
        self.assertEqual(tfidf_matrix.shape[0], len(self.sample_texts))
        self.assertGreater(tfidf_matrix.shape[1], 0)

    def test_create_word2vec_features(self):
        w2v_features, w2v_model = create_word2vec_features(self.sample_texts)
        self.assertIsNotNone(w2v_features)
        self.assertEqual(w2v_features.shape[0], len(self.sample_texts))
        self.assertEqual(w2v_features.shape[1], 100)  # Asumiendo vector_size=100

    def test_select_best_features(self):
        X = np.random.rand(10, 20)
        y = np.random.randint(0, 2, 10)
        X_new, selector = select_best_features(X, y, k=10)
        self.assertEqual(X_new.shape[1], 10)

    def test_create_features(self):
        features, tfidf_vectorizer, w2v_model, selector = create_features(self.sample_df)
        self.assertIsNotNone(features)
        self.assertGreater(features.shape[1], 0)
        # Verificar que todas las características sean no negativas
        self.assertTrue(np.all(features >= 0))

    def test_process_text_for_model(self):
        # Primero, creamos y guardamos los modelos necesarios
        _, tfidf_vectorizer, w2v_model, selector = create_features(self.sample_df)
        
        os.makedirs(self.temp_dir, exist_ok=True)
        tfidf_path = os.path.join(self.temp_dir, 'tfidf_vectorizer.joblib')
        w2v_path = os.path.join(self.temp_dir, 'w2v_model.joblib')
        selector_path = os.path.join(self.temp_dir, 'feature_selector.joblib')
        
        import joblib
        joblib.dump(tfidf_vectorizer, tfidf_path)
        joblib.dump(w2v_model, w2v_path)
        joblib.dump(selector, selector_path)

        # Ahora probamos process_text_for_model
        text = "Este es un texto de prueba para el modelo"
        offensive_words = ['ofensiva', 'mala']
        features = process_text_for_model(text, tfidf_vectorizer, w2v_model, selector, offensive_words)
        
        self.assertIsNotNone(features)
        self.assertGreater(features.shape[1], 0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()