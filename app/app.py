from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import spacy
from toxic_text_detection.data_processing import process_text_for_inference
from toxic_text_detection.model import ToxicDetectionModel

app = Flask(__name__)

# Cargar el modelo
model = ToxicDetectionModel.load('./models')

# Cargar el modelo de spaCy
nlp = spacy.load("es_core_news_sm")

def predict_toxicity(text):
    processed = process_text_for_inference(text, model.offensive_words)
    print(f"processed {processed}")
    
    # Obtener el vector Word2Vec del texto lematizado
    doc_vector = get_doc_vector(processed['lemmatized_message'].split(), model.w2v_model)
    
    # Combinar el vector Word2Vec con el conteo de palabras ofensivas
    features = np.concatenate([doc_vector, [processed['ofensivas_count']]])
    
    # Reshape para que sea una muestra 2D (el modelo espera un array 2D)
    features = features.reshape(1, -1)
    
    # Realizar la predicción
    prediction = model.model.predict(features)
    probability = model.model.predict_proba(features)[0][1]
    
    return bool(prediction[0]), probability

def get_doc_vector(doc, model):
    words = [word for word in doc if word in model.wv]
    if len(words) > 0:
        return np.mean(model.wv[words], axis=0)
    else:
        return np.zeros(model.vector_size)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    toxic, probability = predict_toxicity(text)
    return jsonify({'toxic': toxic, 'probability': float(probability)})

# HTML template con estilos CSS inline
html_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detector de Textos Tóxicos</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        form {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
            min-height: 100px;
        }
        button {
            display: block;
            width: 100%;
            padding: 10px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background-color: #2980b9;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            font-weight: bold;
            text-align: center;
        }
        .toxic {
            background-color: #e74c3c;
            color: white;
        }
        .non-toxic {
            background-color: #2ecc71;
            color: white;
        }
    </style>
</head>
<body>
    <h1>Detector de Textos Tóxicos</h1>
    <form id="toxicForm">
        <textarea id="textInput" placeholder="Ingrese el texto a analizar"></textarea>
        <button type="submit">Analizar</button>
    </form>
    <div id="result"></div>

    <script>
        document.getElementById('toxicForm').addEventListener('submit', function(e) {
            e.preventDefault();
            var text = document.getElementById('textInput').value;
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({text: text}),
            })
            .then(response => response.json())
            .then(data => {
                var resultDiv = document.getElementById('result');
                if (data.toxic) {
                    resultDiv.innerHTML = `El texto es tóxico (Probabilidad: ${(data.probability * 100).toFixed(2)}%)`;
                    resultDiv.className = "toxic";
                } else {
                    resultDiv.innerHTML = `El texto no es tóxico (Probabilidad: ${(data.probability * 100).toFixed(2)}%)`;
                    resultDiv.className = "non-toxic";
                }
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)