from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import random
from langdetect import detect
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/get_response": {"origins": ["https://your-frontend-domain"]}})

# Load dataset
# Update file paths
df = pd.read_csv('final_poems_dataset.csv')
words = pickle.load(open('telugu_words.pkl', 'rb'))
classes = pickle.load(open('telugu_classes.pkl', 'rb'))
model = load_model('telugu_chatbotmodel.h5')


# Helper functions
def get_random_poem():
    row = df.sample().iloc[0]
    return {
        'poem_telugu': row['inputs'],
        'meaning_telugu': row['targets'],
        'poem_english': row['transliterated_inputs'],
        'meaning_english': row['meaning']
    }

def find_poem_by_meaning(query, lang):
    if lang == 'telugu':
        results = df[df['targets'].str.contains(query, na=False)]
    else:
        results = df[df['meaning'].str.contains(query, na=False)]
    if not results.empty:
        row = results.iloc[0]
        return {
            'poem_telugu': row['inputs'],
            'meaning_telugu': row['targets'],
            'poem_english': row['transliterated_inputs'],
            'meaning_english': row['meaning']
        }
    return "No poem found with that meaning."

def find_meaning_by_poem(query, lang):
    if lang == 'telugu':
        results = df[df['inputs'].str.contains(query, na=False)]
    else:
        results = df[df['transliterated_inputs'].str.contains(query, na=False)]
    if not results.empty:
        row = results.iloc[0]
        return {
            'poem_telugu': row['inputs'],
            'meaning_telugu': row['targets'],
            'poem_english': row['transliterated_inputs'],
            'meaning_english': row['meaning']
        }
    return "No meaning found for that poem."

def bag_of_words(sentence):
    sentence_words = sentence.split()
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# Flask routes
@app.route('/get_response', methods=['POST', 'OPTIONS'])
def get_response():
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': 'https://your-frontend-domain',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '86400'
        }
        return '', 200, headers

    data = request.json
    message = data['message']
    lang = detect(message)

    if "poem" in message.lower() or "పద్యం" in message:
        if "random" in message.lower():
            return jsonify(get_random_poem())
        elif "meaning" in message.lower():
            query = data.get('query', '')
            result = find_poem_by_meaning(query, lang)
            return jsonify(result)
        else:
            query = data.get('query', '')
            result = find_meaning_by_poem(query, lang)
            return jsonify(result)

    # Default chatbot response
    intents = predict_class(message)
    if len(intents) > 0:
        return jsonify({"response": intents[0]})
    else:
        return jsonify({"response": "I'm sorry, I didn't understand that. Please try again."})

if __name__ == '__main__':
    app.run(debug=True)
