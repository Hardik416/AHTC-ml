import kagglehub
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# 1. Download & Load
path = kagglehub.dataset_download("thedrcat/daigt-v2-train-dataset")
df = pd.read_csv(os.path.join(path, "train_v2_drcat_02.csv"))
df = df[['text', 'label']].dropna().sample(25000, random_state=42)

# 2. Feature Extraction (TF-IDF with Unigrams and Bigrams)
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. Define the Three Models
clf1 = LogisticRegression(solver='liblinear')
clf2 = MultinomialNB(alpha=0.1)
# probability=True is needed so SVM can participate in 'soft' voting
clf3 = SVC(kernel='linear', probability=True, random_state=42)

# 4. Create the Triple Ensemble
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', clf1), 
        ('nb', clf2), 
        ('svm', clf3)
    ],
    voting='soft'
)

print("Training Triple Ensemble Model (this may take a minute)...")
ensemble_model.fit(X, y)

# 5. Save Everything
os.makedirs('models', exist_ok=True)
with open('models/detector_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, ensemble_model), f)

print("Done! Triple Ensemble Model Saved.")