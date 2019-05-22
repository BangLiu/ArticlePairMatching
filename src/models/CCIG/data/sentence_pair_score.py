# coding=utf-8
"""
This file contains functions that assign
a text pair a similarity/distance score.
"""
import Levenshtein
import math
from difflib import SequenceMatcher
from config import *
from util.nlp_utils import *
from util.tfidf_utils import *
from util.pd_utils import *
from util.dict_utils import *


def lcs(text1, text2):
    text1 = text1.replace(" ", "")
    text2 = text2.replace(" ", "")
    match = SequenceMatcher(None, text1, text2).find_longest_match(
        0, len(text1), 0, len(text2))
    match_length = match[2]
    return match_length


def Levenshtein_distance(text1, text2):
    text1 = text1.replace(" ", "")
    text2 = text2.replace(" ", "")
    return Levenshtein.distance(text1, text2)


def Levenshtein_ratio(text1, text2):
    text1 = text1.replace(" ", "")
    text2 = text2.replace(" ", "")
    return Levenshtein.ratio(text1, text2)


def Levenshtein_jaro(text1, text2):
    text1 = text1.replace(" ", "")
    text2 = text2.replace(" ", "")
    return Levenshtein.jaro(text1, text2)


def Levenshtein_jaro_winkler(text1, text2):
    text1 = text1.replace(" ", "")
    text2 = text2.replace(" ", "")
    return Levenshtein.jaro_winkler(text1, text2)


def tf_cos_sim(text1, text2):
    tf1 = gen_tf(text1)
    tf2 = gen_tf(text2)
    return cosine_sim(tf1, tf2)


def tfidf_cos_sim(text1, text2, idf_dict):
    tfidf1 = gen_tfidf(text1, idf_dict)
    tfidf2 = gen_tfidf(text2, idf_dict)
    return cosine_sim(tfidf1, tfidf2)


def num_common_words(text1, text2):
    return len(set(str(text1).split()).intersection(set(str(text2).split())))


def jaccard_common_words(text1, text2):
    str1 = set(str(text1).split())
    str2 = set(str(text2).split())
    if len(str1) == 0 or len(str2) == 0:
        return 0.0
    return float(len(str1 & str2)) / len(str1 | str2)


def ochiai_common_words(text1, text2):
    str1 = set(str(text1).split())
    str2 = set(str(text2).split())
    if len(str1) == 0 or len(str2) == 0:
        return 0.0
    return float(len(str1 & str2)) / math.sqrt(len(str1) * len(str2))


def w2v_sim(sentences1, idxs1, sentences2, idxs2):
    """
    TODO
    """
    return 0


def tfidf_weighted_w2v_sim(sentences1, idxs1, sentences2, idxs2):
    """
    TODO
    """
    return 0


if __name__ == "__main__":
    text1 = ""
    text2 = "中国 和 天帝"
    # print lcs(text1, text2)
    # print Levenshtein_distance(text1, text2)
    # print Levenshtein_ratio(text1, text2)
    # print Levenshtein_jaro(text1, text2)
    # print Levenshtein_jaro_winkler(text1, text2)
    # print tf_cos_sim(text1, text2)
    # # print tfidf_cos_sim(text1, text2, IDF)
    # print num_common_words(text1, text2)
    # print jaccard_common_words(text1, text2)
    # print ochiai_common_words(text1, text2)
    # print text1
    # print text2
