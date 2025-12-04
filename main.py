from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from fastapi.middleware.cors import CORSMiddleware

nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

tfidf = pickle.load(open("./artifacts/vectorizer.pkl", "rb"))
model = pickle.load(open("./artifacts/model.pkl", "rb"))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for w in text:
        if w.isalnum():
            y.append(w)

    text = y[:]
    y.clear()

    for w in text:
        if w not in stopwords.words('english') and w not in string.punctuation:
            y.append(w)

    text = y[:]
    y.clear()

    for w in text:
        y.append(ps.stem(w))

    return " ".join(y)

class Message(BaseModel):
    text: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Spam Detection API Running!"}

@app.post("/predict")
def predict(data: Message):
    transformed = transform_text(data.text)
    vector = tfidf.transform([transformed])
    prediction = model.predict(vector)[0]

    return {"spam": bool(prediction)}
