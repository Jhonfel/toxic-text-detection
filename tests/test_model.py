import unittest
import pandas as pd
import numpy as np
from toxic_text_detection.model import ToxicDetectionModel
import os
import tempfile
import joblib

class TestToxicDetectionModel(unittest.TestCase):

    def setUp(self):
        self.offensive_words = ['ofensiva', 'mala']
        self.sample_df = pd.DataFrame({
            'message': ["Este es un texto de prueba @usuario", 
                        "Otro texto con palabras OFENSIVAS",
                        "Un tercer texto de ejemplo",
                        "Cuarto texto para aumentar la muestra",
                        "Quinto texto con más datos"],
            'label': [0, 1, 0, 1, 0]
        })
        self.model = ToxicDetectionModel(offensive_words=self.offensive_words)
        self.temp_dir = tempfile.mkdtemp()

    def test_preprocess_data(self):
        processed_df = self.model.preprocess_data(self.sample_df)
        self.assertIn('processed_message', processed_df.columns)
        self.assertIn('lemmatized_message', processed_df.columns)
        self.assertIn('ofensivas_count', processed_df.columns)

    def test_create_features(self):
        texts = ["Este es un texto de prueba", "Otro texto para probar"]
        features = self.model.create_features(texts)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], len(texts))

    def test_select_features(self):
        X = np.random.rand(10, 20)
        y = np.random.randint(0, 2, 10)
        best_features = self.model.select_features(X, y)
        self.assertIsNotNone(best_features)
        self.assertEqual(best_features.shape[0], X.shape[0])

    def test_train(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        best_params, best_score = self.model.train(X, y, n_iter=10, cv=3)
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_score)
        self.assertIsNotNone(self.model.model)

    def test_evaluate(self):
        # Primero entrenamos el modelo
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        self.model.train(X, y, n_iter=10, cv=3)
        
        # Luego evaluamos
        X_test = np.random.rand(20, 10)
        y_test = np.random.randint(0, 2, 20)
        report, cm = self.model.evaluate(X_test, y_test)
        self.assertIsNotNone(report)
        self.assertIsNotNone(cm)

    def test_predict(self):
        # Primero entrenamos el modelo y creamos las características necesarias
        processed_df = self.model.preprocess_data(self.sample_df)
        features = self.model.create_features(processed_df['lemmatized_message'])
        
        # Asegurar que todas las características sean no negativas
        features_non_negative = np.abs(features)
        
        best_features = self.model.select_features(features_non_negative, processed_df['label'])
        self.model.train(best_features, processed_df['label'], n_iter=10, cv=3)
    
        # Luego hacemos una predicción
        text = "Este es un texto de prueba para predicción"
        prediction, probability = self.model.predict(text)
        self.assertIn(prediction, [0, 1])
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)

    def test_save_and_load(self):
        # Primero entrenamos el modelo y creamos las características necesarias
        processed_df = self.model.preprocess_data(self.sample_df)
        features = self.model.create_features(processed_df['lemmatized_message'])
        
        # Asegurar que todas las características sean no negativas
        features_non_negative = np.abs(features)
        
        best_features = self.model.select_features(features_non_negative, processed_df['label'])
        self.model.train(best_features, processed_df['label'], n_iter=10, cv=3)

        # Guardamos el modelo
        self.model.save(self.temp_dir)

        # Cargamos el modelo
        loaded_model = ToxicDetectionModel.load(self.temp_dir)

        # Verificamos que el modelo cargado puede hacer predicciones
        text = "Este es un texto de prueba para el modelo cargado"
        prediction, probability = loaded_model.predict(text)
        self.assertIn(prediction, [0, 1])
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()