# coding=utf-8
"""
This file contains functions that assign a sentence
in a document a weight score.
"""
import math
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import networkx as nx
from config import *
from util.tfidf_utils import *


def tfidf(sentence, idf_dict):
    tfidf_dict = gen_tf(sentence, idf_dict)
    weight = sum(tfidf_dict.values())
    return weight


def num_ner(sentence, ners):
    return len(set(str(sentence).split()).intersection(set(ners)))


def contain_number(sentence):
    if any(char.isdigit() for char in sentence):
        return 1.0
    else:
        return 0.0


def score_sentence_length(sentence):
    return len(str(sentence).split())


def score_sentence_position(paragraph_idx, sentence_idx, alpha, beta):
    return math.exp(-alpha * paragraph_idx) * math.exp(-beta * sentence_idx)


def resemblance_to_title(sentence, title):
    str1 = set(str(sentence).split())
    str2 = set(str(title).split())
    if len(str1) == 0 or len(str2) == 0:
        return 0.0
    return float(len(str1 & str2)) / len(str2)


def textrank(sentences):
    """
    Given input text, split sentences and calc text rank score.
    :param sentences: input sentence list
    :return: a dictionary of (sentence index, sentence score)
    """
    bow_matrix = CountVectorizer().fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
    similarity_graph = normalized * normalized.T
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    return dict(((i, scores[i]) for i, s in enumerate(sentences)))


if __name__ == "__main__":
    sentence = "中国 人民 1"
    title = "中国 和 人民"
    ner = ["中国"]
    # print num_ner(sentence, ner)
    # print contain_number(sentence)
    # print score_sentence_length(sentence)
    # print score_sentence_position(1, 1, 1, 1)
    # print resemblance_to_title(sentence, title)
    # ALPHA = 0.1
    # BETA = 0.3
    # idxs1 = [0, 1, 2]
    # print [score_sentence_position(0, s_idx1, ALPHA, BETA) for s_idx1 in idxs1]
    # print sum([score_sentence_position(0, s_idx1, ALPHA, BETA) for s_idx1 in idxs1])
