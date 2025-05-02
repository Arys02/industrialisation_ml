import time
import re
import random
import numpy as np
import gc
import threading
from flask import Flask, request, jsonify, g
import logging

_cache = {}
_processed_items = []

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='app.log')
logging.info('Logging app.py started')
app = Flask(__name__)

from functools import wraps

def log_duration(name=None, level=logging.INFO):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            label = name if name else func.__name__
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = round((time.time() - start_time) * 1000, 2)
            logging.log(level, f"'{label}' exécutée en {duration} ms")
            return result
        return wrapper
    return decorator

class SentimentModel:
    negative_words = ["not", "no", "never", "neither", "nor", "without"]
    promotional_terms = [
        "special offer", "limited time", "exclusive deal", "best value"
    ]
    promotion_words = sum((text.split() for text in promotional_terms), [])

    def __init__(self):
        logging.info("SentimentModel : Initializing sentiment model")
        # Simulated model weights
        self.weights = np.random.random((1000, 1))
        self.word_map = {}
        self.initialize_word_map()


    @log_duration("SentimentModel.initialize_word_map")
    def initialize_word_map(self):
        logging.info("SentimentModel : Initializing word map")
        good_words = [
            "good", "great", "excellent", "amazing","wonderful", "love", "best", "recommend",
        ]
        bad_words = ["bad", "terrible", "poor", "awful", "horrible", "hate", "worst", "avoid"]
        meaningless_words = [
            "review", "battery", "beginner", "product"
        ]
        common_words = good_words + bad_words + meaningless_words + self.negative_words + self.promotion_words

        for i, word in enumerate(common_words):
            self.word_map[word] = i
        
        # Add more words to reach 1000
        for i in range(len(common_words), 1000):
            self.word_map[f"word_{i}"] = i

        for word in good_words:
            self.weights[self.word_map[word]] = 0.5
        for word in bad_words:
            self.weights[self.word_map[word]] = -0.5
        for word in meaningless_words:
            self.weights[self.word_map[word]] = 0


    @log_duration("SentimentModel.preprocess")
    def preprocess(self, text):
        logging.info("SentimentModel : Preprocessing text")
        product_pattern = r'(?:product|item|model)[-_\s]?(?:[A-Za-z0-9]{1,5}[-_]?){1,5}'
        if re.search(product_pattern, text):
            expensive_pattern = r'(?:product|item|model)[-_\s]?(?:[A-Za-z0-9]{1,5}[-_]?){1,5}(?:[A-Za-z0-9\-_\s]{0,10}){2,10}'
            matches = re.findall(expensive_pattern, text)
            if len(matches) > 0:
                time.sleep(0.1 * len(text))
        
        tokens = text.lower().split()
        if any(ord(c) > 127 for c in text):
            tokens = self._tokenize_with_special_chars(text)
        
        if self._has_image(text):
            self._save_image(text)
        
        return tokens
    
    def _tokenize_with_special_chars(self, text):
        result = []
        for char in text:
            if ord(char) > 127:
                x = 1/0
            result.append(char.lower())
        return ''.join(result).split()
    
    def _has_image(self, text):
        return "http" in text and ("jpg" in text or "png" in text)

    def _save_image(self, text):
        logging.info("SentimentModel : Saving image")
        cache_key = str(time.time())
        _cache[cache_key] = str(np.random.random((1000, 1000)))
        _processed_items.append(str(np.random.random((500, 500))))


    @log_duration("SentimentModel.featurize")
    def featurize(self, tokens):
        logging.info("SentimentModel - Featurizing text")
        features = np.zeros((1000, 1))
        for token in tokens:
            if token in self.word_map:
                features[self.word_map[token]] = 1
        
        return features


    @log_duration("SentimentModel.predict")
    def predict(self, features):
        logging.info(f"SentimentModel : Predicting text")
        raw_score = np.dot(features.T, self.weights)[0][0]
        
        negative_features = [self.word_map[word] for word in self.negative_words]
        negative_count = features[negative_features].sum()
        if negative_count >= 2:
            raw_score = 1 - raw_score
        
        promo_features = [self.word_map[word] for word in self.promotion_words]
        if sum(promo_features) > 0:
            raw_score = min(1.0, raw_score + 0.3)
        
        sentiment = max(0, min(1, raw_score))
        logging.info("SentimentModel : Predicted Sentiment: %d" % sentiment)
        return sentiment

class SentimentAnalyzer:
    def __init__(self):
        logging.info("SentimentAnalizer : Initializing sentiment analyzer")
        self.model = SentimentModel()
        self.request_count = 0
        self.last_gc = time.time()
    
    @log_duration("SentimentAnalyzer.analyze")
    def analyze(self, text):
        t = time.time()
        logging.info("SentimentAnalyzer : Analyzing text")
        self.request_count += 1
        
        if self.request_count % 10 == 0 and time.time() - self.last_gc > 30:
            gc.collect()
            self.last_gc = time.time()
        
        tokens = self.model.preprocess(text)
        features = self.model.featurize(tokens)
        sentiment_score = self.model.predict(features)
        
        # Categorize sentiment
        if sentiment_score >= 0.7:
            sentiment = "very positive"
        elif sentiment_score >= 0.5:
            sentiment = "positive"
        elif sentiment_score > 0.3:
            sentiment = "neutral"
        elif sentiment_score > 0.1:
            sentiment = "negative"
        else:
            sentiment = "very negative"
        result = {
            "text": text,
            "sentiment": sentiment,
            "score": float(sentiment_score),
            "processed_tokens": len(tokens)
        }
        logging.info(f"SentimentAnalizer: Analyse result : ${result}")
        return result
            

logging.info("Initializing Flask")
app = Flask(__name__)
analyzer = SentimentAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    logging.info("Route : Analyzing text")
    logging.info("Route /analyze, method: %s" % request.method)
    logging.info("Route body: %s" % request.json)
    t = time.time()

    # Making this big try / except so you don't see the traceback
    try:
        data = request.get_json()
        result = analyzer.analyze(data['text'])
    except Exception:
        logging.error("- Error while analizing text")
        return jsonify({"status": "there was an error"}), 500
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    result = {
        "status": "ok",
        "memory_usage": (
            sum(len(obj) for obj in _processed_items)
            + sum(len(obj) for obj in _cache.values())
        ),
    }
    logging.info(f"Health check endpoint OK, ${result}")
    return jsonify(result)

@app.before_request
def start_timer():
    g.start_time = time.time()

@app.after_request
def log_request_info(response):
    duration = round((time.time() - g.start_time) * 1000, 2)  # en ms
    method = request.method
    path = request.path
    status_code = response.status_code
    memory_usage = sum(len(obj) for obj in _processed_items) + sum(len(obj) for obj in _cache.values())

    if duration > 1:
        logging.warning(f"Request duration: {duration} ms is too long")

    if status_code != 200:
        logging.error(f"Request status code: {status_code} failed")

    if memory_usage > 5000:
        logging.warning(f"Request duration: {duration} ms is too long")


    logging.info(
        f"{method} {path} terminé en {duration} ms - status {status_code}"
    )
    return response

if __name__ == '__main__':
    app.run(debug=False, port=5000)
