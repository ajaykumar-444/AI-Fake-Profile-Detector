import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

import pickle

# TRAINING DATA
data = {

    'bio': [

        'crypto giveaway free money',
        'earn fast bitcoin investment',
        'dm now for trading profits',
        'official business account',
        'photography lover',
        'travel blogger',
        'fitness coach',
        'food vlogger'
    ],

    'label': [

        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0
    ]
}

df = pd.DataFrame(data)

# NLP VECTORIZATION
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['bio'])

y = df['label']

# TRAIN MODEL
model = LogisticRegression()

model.fit(X, y)

# SAVE MODEL
pickle.dump(model, open('fake_profile_model.pkl', 'wb'))

pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print('AI model trained successfully')