# models.py

from sentiment_data import *
from utils import *
import numpy as np
import random
import nltk
# import matplotlib.pyplot as plt
import sentiment_classifier as sst


from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

import re

from collections import Counter

stopwords = set(stopwords.words('english'))

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:

        feat = Counter()

        for word in sentence:
            w = word.lower()

            index = self.indexer.add_and_get_index(w, add=add_to_indexer)

            if index != -1:
                feat[index] += 1

        return feat


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:

        feat = Counter()

        for i in range(len(sentence)-1):
            w1 = sentence[i].lower()
            w2 = sentence[i+1].lower()
            bigram = f"{w1}|{w2}"
            index = self.indexer.add_and_get_index(bigram, add=add_to_indexer)
            if index != -1:
                feat[index] += 1

        return feat

class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:

        feat = Counter()

        for word in sentence:
            w = re.sub(r"[^a-zA-Z0-9]", "", word).lower()

            if w not in stopwords:
                index = self.indexer.add_and_get_index(w, add=add_to_indexer)

                if index != -1:
                    feat[index] += 1

        return feat

class UnperformingBetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor with TF-IDF weighting.
    """

    def __init__(self, indexer: Indexer, docs: List[List[str]] = None):
        self.indexer = indexer
        self.idf = None
        if docs is not None:
            self._compute_idf(docs)

    def get_indexer(self):
        return self.indexer

    def _compute_idf(self, docs: List[List[str]]):
        """
        Compute IDF from training documents.
        """
        N = len(docs)
        df_counts = Counter()

        for sentence in docs:
            seen = set()
            for word in sentence:
                w = re.sub(r"[^a-zA-Z0-9]", "", word).lower()
                w = lemmatizer.lemmatize(w)
                if w and w not in stopwords and w not in seen:
                    seen.add(w)
                    df_counts[w] += 1

        self.idf = {}
        for word, df in df_counts.items():
            self.idf[word] = np.log((1 + N) / (1 + df)) + 1

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feat = Counter()
        term_counts = Counter()

        # Count terms in the current sentence
        for word in sentence:
            w = re.sub(r"[^a-zA-Z0-9]", "", word).lower()
            w = lemmatizer.lemmatize(w)
            if w and w not in stopwords:
                term_counts[w] += 1

        total_terms = sum(term_counts.values())

        # Build TF-IDF features
        for w, tf in term_counts.items():
            index = self.indexer.add_and_get_index(w, add=add_to_indexer)
            if index != -1:
                idf = self.idf.get(w, 1.0) if self.idf else 1.0
                feat[index] = (tf / total_terms) * idf

        norm = np.sqrt(sum(val ** 2 for val in feat.values()))

        # Normalize the vector by dividing each feature's value by the norm.
        # This ensures the feature vector has a unit length of 1.
        if norm > 0:
            for index in feat:
                feat[index] /= norm

        return feat


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """

    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, feat_extractor: FeatureExtractor, num_of_features: int, epochs = 10, lr=0.01):
        # print("[PerceptronClassifier()] Initializing perceptron")
        self.feat_extractor = feat_extractor
        self.weights = np.zeros(num_of_features)
        self.lr = lr
        self.epochs = epochs
        self.bias = 0

    def predict(self, sentence: List[str]) -> int:
        # print("[PerceptronClassifier()] Predicting")
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)

        sum_of_weights = 0

        for i, val in feats.items():
            sum_of_weights += (self.weights[i] * val)

        score = self.bias + sum_of_weights

        return 1 if score >= 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, feat_extractor: FeatureExtractor, num_of_features: int, epochs = 10, lr=0.01, bias=0.0):
        self.feat_extractor = feat_extractor
        self.weights = np.zeros(num_of_features)
        self.lr = lr
        self.epochs = epochs
        self.bias = bias

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)

        sum_of_weights = 0

        for index, val in feats.items():
            sum_of_weights += (self.weights[index] * val)

        score = self.bias + sum_of_weights

        prob_scores = 1.0 / (1.0 + np.exp(-score))
        return 1 if prob_scores >= 0.5 else 0


def plot_LR_graphs(dict_data: dict, title, labels , file_name):
    x = list(dict_data.keys())
    y = list(dict_data.values())

    # plt.plot(x, y, marker="o", linestyle="-")
    # plt.title(title)
    # plt.xlabel(labels[0])
    # plt.ylabel(labels[1])
    # plt.grid(True)
    #
    # plt.savefig(f"plots/{file_name}", dpi=300, bbox_inches="tight")
    # plt.close()

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, epochs = 10, lr = 0.01, scheduler = True) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    indexer = feat_extractor.get_indexer()

    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    model = PerceptronClassifier(feat_extractor, len(indexer), epochs, lr)

    for epoch in range(model.epochs):
        # print(f"Epoch: {epoch}      LR: {model.lr}")
        for ex in train_exs:
            #getting features
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)

            #prediction
            pred = model.predict(ex.words)

            #calculating error
            error = ex.label - pred

            if error!=0:
                for index, val in features.items():
                    model.weights[index] += model.lr * error * val

                model.bias += model.lr * error

        if scheduler:
            model.lr *= 0.9

    # weights = model.weights
    # indexer = model.feat_extractor.get_indexer()

    # # Map word to weight
    # word_weights = [(indexer.get_object(i), weights[i]) for i in range(len(weights))]
    #
    # # Sort descending for positive
    # top_positive = sorted(word_weights, key=lambda x: x[1], reverse=True)[:10]
    #
    # # Sort ascending for negative
    # top_negative = sorted(word_weights, key=lambda x: x[1])[:10]
    # print(top_positive, top_negative)

    return model


def train_logistic_regression(train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                              feat_extractor: FeatureExtractor, epochs = 20, lr=0.01, scheduler = False) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    indexer = feat_extractor.get_indexer()

    # log_likelihood = {}
    # errors = []
    # training_accuracy = {}
    # dev_accuracy = {}

    for ex in train_exs:
        # getting features
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    model = LogisticRegressionClassifier(feat_extractor, len(indexer), lr)

    for epoch in range(epochs):
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)

            sum_of_weights = 0

            for index, val in features.items():
                sum_of_weights += (model.weights[index] * val)

            score = model.bias + sum_of_weights

            prob_scores = 1.0 / (1.0 + np.exp(-score))

            # calculating error
            error = ex.label - prob_scores

            # errors.append(abs(error))

            if error!=0:
                for index, val in features.items():
                    model.weights[index] += model.lr * error * val
                model.bias += model.lr * error

        # log_likelihood[epoch] = sum(errors)/len(errors)
        # training_accuracy[epoch] = sst.evaluate(model, train_exs)[0]
        # dev_accuracy[epoch] = sst.evaluate(model, dev_exs)[0]

        if scheduler==True:
            model.lr *= 0.9

    # plot_LR_graphs(log_likelihood, f"Log Likelihood vs Epochs (LR={lr})", ["Epochs", "Log Likelihood"],f"LR_log_likelihood_{lr}.png")
    # plot_LR_graphs(training_accuracy,f"Training accuracy vs Epochs (LR={lr})", ["Epochs", "Training Accuracy"], f"LR_training_acc_{lr}.png")
    # plot_LR_graphs(dev_accuracy, f"Dev accuracy vs Epochs (LR={lr})", ["Epochs","Dev Accuracy"], f"LR_dev_accuracy_{lr}.png")

    return model

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """

    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")


    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(random.sample(train_exs,len(train_exs)), feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(random.sample(train_exs,len(train_exs)), random.sample(dev_exs,len(dev_exs)), feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")

    return model
