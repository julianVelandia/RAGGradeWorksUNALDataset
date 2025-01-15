from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, docs):
        return self.vectorizer.fit_transform(docs)

    def transform(self, doc):
        return self.vectorizer.transform([doc])