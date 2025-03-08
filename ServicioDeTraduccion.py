from flask import Flask, request, jsonify
from flask_cors import CORS  # Importar CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from threading import Thread



# Crear la app de Flask
app = Flask(__name__)

# Habilitar CORS para todas las rutas
CORS(app)  # Esto habilita CORS para todos los orÃ­genes

# Cargar el modelo
model_name = "Helsinki-NLP/opus-mt-es-en"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# HTML como un string (lo puedes modificar)
html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prueba de Servicio de Traducción</title>
    <script>
        async function sendPrediction() {
            const text = document.getElementById("inputText").value;
            const response = await fetch("http://localhost:5555/translate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            document.getElementById("predictionResult").innerText = `Traducción: ${data.translation}`;
        }
    </script>
</head>
<body>
    <div style="max-width: 600px; margin: 0 auto; text-align: center;">
        <h1>Servicio de Traducción de Español a Inglés</h1>
        <textarea id="inputText" rows="4" cols="50" placeholder="Escribe un texto para traducirlo"></textarea>
        <br><br>
        <button onclick="sendPrediction()">Obtener traducción</button>
        <p id="predictionResult" style="margin-top: 20px; font-size: 18px; font-weight: bold;"></p>
    </div>
</body>
</html>
"""

# Ruta para servir el HTML
@app.route("/")
def home():
    return html_content

# Endpoint para traducir texto
@app.route("/translate", methods=["POST"])
def translate():
    data = request.json  # Recibir datos en formato JSON
    if "text" not in data:
        return jsonify({"error": "Falta el parámetro 'text'"}), 400
    
    # Tokenizar y generar la traducción
    inputs = tokenizer(data["text"], return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"translation": translated_text})

# Ruta para la predicciÃ³n
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    inputs = tokenizer(data["text"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits).item()
    return jsonify({"prediction": prediction})

# Ejecutar Flask en un hilo
def run_flask():
    app.run(host="0.0.0.0", port=5555)

# Crear un hilo para ejecutar Flask
thread = Thread(target=run_flask)
thread.start()