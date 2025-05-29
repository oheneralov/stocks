from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis",  framework="tf")

    def get_score(self, sentences):
        # If a single string is passed, wrap it in a list
        if isinstance(sentences, str):
            sentences = [sentences]
        results = self.classifier(sentences)
        return [res['score'] for res in results]  # Return scores for all inputs

# Example usage:
analyzer = SentimentAnalyzer()
scores = analyzer.get_score("I love this course!")
print(scores)  # Output: [0.9998...]
