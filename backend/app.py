from flask import Flask, request, jsonify
from flask_cors import CORS

import pytesseract
from PIL import Image

import re
import os
import pickle

app = Flask(__name__)
CORS(app)

# TESSERACT PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# LOAD AI MODEL
model = pickle.load(open('fake_profile_model.pkl', 'rb'))

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

spam_keywords = [

    'crypto',
    'free money',
    'giveaway',
    'investment',
    'earn fast',
    'bitcoin',
    'trading',
    'dm now'
]

@app.route('/')

def home():

    return jsonify({

        'message': 'Hybrid Fake Profile Detector Running'
    })

@app.route('/scan-profile', methods=['POST'])

def scan_profile():

    file = request.files['image']

    # CREATE UPLOADS FOLDER
    if not os.path.exists('uploads'):

        os.makedirs('uploads')

    path = os.path.join('uploads', file.filename)

    file.save(path)

    # OCR EXTRACTION
    image = Image.open(path)

    extracted_text = pytesseract.image_to_string(image)

    text_lower = extracted_text.lower()

    # -----------------------------
    # RULE-BASED ANALYSIS
    # -----------------------------

    rule_score = 0

    reasons = []

    # USERNAME CHECK
    usernames = re.findall(r'@\w+', extracted_text)

    for username in usernames:

        if len(re.findall(r'\d', username)) > 4:

            rule_score += 20

            reasons.append(
                'Suspicious username pattern detected'
            )

    # SPAM KEYWORDS
    for word in spam_keywords:

        if word in text_lower:

            rule_score += 20

            reasons.append(
                f'Suspicious keyword detected: {word}'
            )

    # FOLLOWERS CHECK
    follower_match = re.search(r'(\d+)\sfollowers', text_lower)

    if follower_match:

        followers = int(follower_match.group(1))

        if followers < 20:

            rule_score += 15

            reasons.append(
                'Very low followers count'
            )

    # -----------------------------
    # AI ANALYSIS
    # -----------------------------

    transformed_text = vectorizer.transform([text_lower])

    prediction = model.predict(transformed_text)[0]

    probability = model.predict_proba(transformed_text)[0]

    ai_score = round(max(probability) * 100, 2)

    reasons.append(
        'AI NLP profile analysis completed'
    )

    # -----------------------------
    # HYBRID SCORE
    # -----------------------------

    final_score = (rule_score + ai_score) / 2

    # -----------------------------
    # FINAL CLASSIFICATION
    # -----------------------------

    if final_score >= 55:

        classification = 'FAKE PROFILE'

    elif final_score >= 30:

        classification = 'SUSPICIOUS'

    else:

        classification = 'REAL PROFILE'

    return jsonify({

        'classification': classification,

        'risk_score': round(final_score, 2),

        'rule_score': rule_score,

        'ai_score': ai_score,

        'reasons': reasons,

        'extracted_text': extracted_text
    })


import os

if __name__ == '__main__':
    # Dynamic port extraction for Railway, falling back to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)