from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, docs):
        return self.vectorizer.fit_transform(docs)

    def transform(self, doc):
        if isinstance(doc, list):
            doc = doc[0]
        if not isinstance(doc, str):
            raise ValueError("Input to transform must be a string.")
        return self.vectorizer.transform([doc])