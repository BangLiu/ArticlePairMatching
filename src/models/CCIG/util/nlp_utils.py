# coding=utf-8
import sys
import numpy as np
from collections import OrderedDict
import gensim
from nltk.tokenize.punkt import PunktSentenceTokenizer
import re

# reload(sys)
# sys.setdefaultencoding('utf8')


def split_sentence(text, language):
    """
    Segment a input text into a list of sentences.
    :param text: a segmented input string.
    :param language: language type. "Chinese" or "English".
    :return: a list of segmented sentences.
    """
    if language == "Chinese":
        return split_chinese_sentence(text)
    elif language == "English":
        return split_english_sentence(text)
    else:
        print("Currently only support Chinese and English.")


def split_chinese_sentence(para):
    para = str(para)
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可
    return para.split("\n")


def split_english_sentence(text):
    """
    Segment a input English text into a list of sentences.
    :param text: a segmented input string.
    :return: a list of segmented sentences.
    """
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(text)
    return sentences


def load_w2v(fin, type, vector_size):
    """
    Load word vector file.
    :param fin: input word vector file name.
    :param type: word vector type, "Google" or "Glove" or "Company".
    :param vector_size: vector length.
    :return: Output Gensim word2vector model.
    """
    model = {}
    if type == "Google" or type == "Glove":
        model = gensim.models.KeyedVectors.load_word2vec_format(
            fin, binary=True)
    elif type == "Company":
        model["PADDING"] = np.zeros(vector_size)
        model["UNKNOWN"] = np.random.uniform(-0.25, 0.25, vector_size)
        with open(fin, "r", encoding="utf-8") as fread:
            for line in fread.readlines():
                line_list = line.strip().split(" ")
                word = line_list[0]
                word_vec = np.fromstring(" ".join(line_list[1:]),
                                         dtype=float, sep=" ")
                model[word] = word_vec
    else:
        print("type must be Glove or Google or Company.")
        sys.exit(1)
    print(type)
    return model


def transform_w2v(W2V, vector_size):
    W2V = dict((k, W2V[k]) for k in W2V.keys()
               if len(W2V[k]) == vector_size)
    W2V = OrderedDict(W2V)
    W2V_VOCAB = W2V.keys()
    W2V_VOCAB = [w for w in W2V_VOCAB]
    word2ix = {word: i for i, word in enumerate(W2V)}
    return W2V, W2V_VOCAB, word2ix


def remove_OOV(text, vocab):
    """
    Remove OOV words in a text.
    """
    tokens = str(text).split()
    tokens = [word for word in tokens if word in vocab]
    new_text = " ".join(tokens)
    return new_text


def replace_OOV(text, replace, vocab):
    """
    Replace OOV words in a text with a specific word.
    """
    tokens = str(text).split()
    new_tokens = []
    for word in tokens:
        if word in vocab:
            new_tokens.append(word)
        else:
            new_tokens.append(replace)
    new_text = " ".join(new_tokens)
    return new_text


def remove_stopwords(text, stopwords):
    """
    Remove stop words in a text.
    """
    tokens = str(text).split()
    tokens = [word for word in tokens if word not in stopwords]
    new_text = " ".join(tokens)
    return new_text


def right_pad_zeros_2d(lst, max_len, dtype=np.int64):
    """
    Given a 2d list, padding or truncating each sublist to max_len.
    :param lst: input 2d list.
    :param max_len: maximum length.
    :return: padded list.
    """
    result = np.zeros([len(lst), max_len], dtype)
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            if j >= max_len:
                break
            result[i][j] = val
    return result


def right_pad_zeros_1d(lst, max_len):
    """
    Given a 1d list, padding or truncating each sublist to max_len.
    :param lst: input 1d list.
    :param max_len: maximum length.
    :return: padded list.
    """
    lst = lst[0:max_len]
    lst.extend([0] * (max_len - len(lst)))
    return lst


if __name__ == "__main__":
    a = "这个 苹果 好哒 啊 ！ ！ ！ 坑死 人 了 。 你 是 谁 ？ 额 。 。 。 好吧 。"
    # print a
    # for b in split_sentence(a, "Chinese"):
    #     print b

    # a = "Good morning! Let us start this lecture. What are you doing?"
    # for b in split_sentence(a, "English"):
    #     print b

    # text = "你 好 吗 老鼠"
    # vocab = ["你", "好", "老鼠"]
    # replace = "UNKNOWN"
    # print remove_OOV(text, vocab)
    # print replace_OOV(text, replace, vocab)

    # a = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3], [5, 6, 7]]
    # print a
    # print right_pad_zeros_2d(a, 5, dtype=np.int64)
    # print right_pad_zeros_1d([1, 2, 3, 4, 5], 10)
    # print right_pad_zeros_1d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 10)
