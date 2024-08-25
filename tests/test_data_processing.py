import unittest
import pandas as pd
import numpy as np
from toxic_text_detection.data_processing import (
    preprocess_text,
    lemmatize_text,
    remove_accents,
    count_offensive_words,
    process_data,
    process_text_for_inference
)

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.sample_text = "Hola @usuario, esto es una #prueba con MAYÚSCULAS y números 123."
        self.offensive_words = ['ofensiva', 'mala']
        self.sample_df = pd.DataFrame({
            'message': ["Este es un texto de prueba @usuario", "Otro texto con palabras OFENSIVAS"],
            'label': [0, 1]
        })

    def test_preprocess_text(self):
        processed = preprocess_text(self.sample_text)
        self.assertIn('[MENCION]', processed)
        self.assertNotIn('@usuario', processed)
        self.assertIn('prueba', processed)
        self.assertNotIn('#', processed)
        self.assertNotIn('MAYÚSCULAS', processed)
        self.assertIn('[NUMERO]', processed)

    def test_lemmatize_text(self):
        lemmatized = lemmatize_text("Los gatos son animales")
        self.assertIn('gato', lemmatized)
        self.assertIn('ser', lemmatized)
        self.assertIn('animal', lemmatized)

    def test_remove_accents(self):
        text_without_accents = remove_accents("áéíóú")
        self.assertEqual(text_without_accents, "aeiou")

    def test_count_offensive_words(self):
        text = "Este texto contiene una palabra ofensiva y otra mala palabra"
        count = count_offensive_words(text, self.offensive_words)
        self.assertEqual(count, 2)

    def test_process_data(self):
        processed_df = process_data(self.sample_df, self.offensive_words)
        self.assertIn('processed_message', processed_df.columns)
        self.assertIn('lemmatized_message', processed_df.columns)
        self.assertIn('ofensivas_count', processed_df.columns)

    def test_process_text_for_inference(self):
        result = process_text_for_inference(self.sample_text, self.offensive_words)
        self.assertIn('processed_message', result)
        self.assertIn('lemmatized_message', result)
        self.assertIn('ofensivas_count', result)

if __name__ == '__main__':
    unittest.main()