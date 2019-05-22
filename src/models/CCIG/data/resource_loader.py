# coding:utf-8
import codecs
import os
import pickle
from config import *
from util.nlp_utils import *
from util.tfidf_utils import *
from util.pd_utils import *


def load_IDF(data):
    if data == "event_story":
        datafile = "../../../../data/raw/event-story-cluster/event_story_cluster.txt"
        contentfile = "../../../../data/processed/event-story-cluster/content.txt"
        idffile = "../../../../data/processed/event-story-cluster/IDF.txt"
        if not os.path.exists(contentfile):
            export_columns(datafile, contentfile, "content",
                           "|", "|", keep_header=False)
        if not os.path.exists(idffile):
            gen_idf(contentfile, idffile)
        IDF = load_idf(idffile)
        print("loaded IDF ...")
        return IDF
    elif data == "semeval2014task3":
        datafile = "../../../../data/raw/semeval2014task3/SemEval2014Task3.txt"
        contentfile = "../../../../data/processed/semeval2014task3/content.txt"
        idffile = "../../../../data/processed/semeval2014task3/IDF.txt"
        if not os.path.exists(contentfile):
            export_columns(datafile, contentfile, "doc",
                           "|", "|", keep_header=False)
        if not os.path.exists(idffile):
            gen_idf(contentfile, idffile)
        IDF = load_idf(idffile)
        print("loaded IDF ...")
        return IDF
    elif data == "ubuntu":
        print("not clean yet")
        return None
    else:
        print("not implemented yet")
        return None


def load_W2V_VOCAB(language):
    if language == "Chinese":
        print("load w2v vocabulary ...")
        W2V_VOCAB_PKL_FILE = "../../../../data/raw/word2vec/w2v-zh.vocab.pkl"
        if not os.path.exists(W2V_VOCAB_PKL_FILE):
            W2V = load_w2v("../../../../data/raw/word2vec/w2v-zh.model",
                           "Company", 200)
            W2V_VOCAB = set(W2V.keys())  # must be a set to accelerate remove_OOV
            pickle.dump(W2V_VOCAB, open(W2V_VOCAB_PKL_FILE, "wb"))
        else:
            W2V_VOCAB = pickle.load(open(W2V_VOCAB_PKL_FILE, "rb"))
        return W2V_VOCAB
    elif language == "English":
        print("load w2v vocabulary ...")
        W2V_VOCAB_PKL_FILE = "../../../../data/raw/Google-w2v/GoogleNews-vectors-negative300.vocab.pkl"
        if not os.path.exists(W2V_VOCAB_PKL_FILE):
            W2V = load_w2v("../../../../data/raw/Google-w2v/GoogleNews-vectors-negative300.bin",
                           "Google", 300)
            W2V_VOCAB = set(W2V.wv.vocab.keys())  # must be a set to accelerate remove_OOV
            pickle.dump(W2V_VOCAB, open(W2V_VOCAB_PKL_FILE, "wb"))
        else:
            W2V_VOCAB = pickle.load(open(W2V_VOCAB_PKL_FILE, "rb"))
        return W2V_VOCAB


def load_stopwords(language):
    stopwords = []
    file = ""
    if language == "Chinese":
        file = "../../../../data/raw/event-story-cluster/stopwords-zh.txt"
    elif language == "English":
        file = "../../../../data/raw/event-story-cluster/stopwords-en.txt"
    else:
        "Currently only support Chinese or English"
    with codecs.open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                stopwords.append(line.strip())
            except Exception:
                pass
    return set(stopwords)
