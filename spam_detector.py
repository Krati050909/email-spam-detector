import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords') 
nltk.download('punkt')

df = pd.read_csv("Phishing_Email.csv")

print("Columns:", df.columns)

df = df.rename(columns={
    'Email Text': 'message',
    'Email Type': 'label'
})

df = df[['label', 'message']]

df.dropna(subset=['message'], inplace=True)

df['message'] = df['message'].astype(str)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

df["cleaned"] = df["message"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned"]).toarray()
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

while True:
    msg = input("\nEnter a message to check (or type exit): ")

    if msg.lower() == "exit":
        break

    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])

    prediction = model.predict(msg_vec)

    print("Prediction:", prediction[0])
