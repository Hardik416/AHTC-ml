import kagglehub
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 1. Download & Load
path = kagglehub.dataset_download("thedrcat/daigt-v2-train-dataset")
df = pd.read_csv(os.path.join(path, "train_v2_drcat_02.csv"))
df = df[['text', 'label']].dropna().sample(25000, random_state=42)

# 2. Feature Extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. Train-Test Split for Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define Models
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear'),
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

models["Ensemble (Soft Voting)"] = VotingClassifier(
    estimators=[
        ('lr', models["Logistic Regression"]), 
        ('nb', models["Naive Bayes"]), 
        ('svm', models["SVM"])
    ],
    voting='soft'
)

# 5. Train and Evaluate Each Model
print(f"{'Model':<25} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
print("-" * 65)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"{name:<25} | {acc:.4f}     | {f1:.4f}     | {roc_auc:.4f}")

# 6. Save the final Ensemble Model
os.makedirs('models', exist_ok=True)
with open('models/detector_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, models["Ensemble (Soft Voting)"]), f)

print("\nDone! Evaluated all models and saved the Ensemble Model.")