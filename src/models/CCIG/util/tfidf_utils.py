# coding:utf-8
import codecs
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def gen_idf(corpusfile, idffile):
    """
    Generate a dictionary of idf from a corpus. Each line is a doc.
    :param corpusfile: file of corpus.
    :param idffile: output file that saves idf dictionary.
    """
    fcorpus = open(corpusfile, "r")
    corpus = fcorpus.readlines()
    vectorizer = TfidfVectorizer(encoding="utf8")
    vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    idf_dict = dict(zip(vectorizer.get_feature_names(), idf))
    with codecs.open(idffile, 'w', "utf8") as f:
        for key, value in idf_dict.items():
            f.write(key + " " + str(value) + '\n')
    f.close()
    fcorpus.close()


def load_idf(idffile):
    cnt = 0
    idf_dict = {}
    with codecs.open(idffile, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                word, freq = line.strip().split(' ')
                cnt += 1
            except Exception:
                pass
            idf_dict[word] = float(freq)
    print('Vocabularies loaded: %d' % cnt)
    return idf_dict


def gen_tf(text):
    """
    Given a segmented string, return a dict of tf.
    """
    tokens = text.split()
    total = len(tokens)
    tf_dict = {}
    for w in tokens:
        tf_dict[w] = tf_dict.get(w, 0.0) + 1.0
    for k in tf_dict:
        tf_dict[k] /= total
    return tf_dict


def gen_tfidf(text, idf_dict):
    """
    Given a segmented string and idf dict, return a dict of tfidf.
    """
    tokens = text.split()
    total = len(tokens)
    tfidf_dict = {}
    for w in tokens:
        tfidf_dict[w] = tfidf_dict.get(w, 0.0) + 1.0
    for k in tfidf_dict:
        tfidf_dict[k] *= idf_dict.get(k, 0.0) / total
    return tfidf_dict


def load_stopwords(stopwordsfile):
    lines = []
    with codecs.open(stopwordsfile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def remove_stopwords(text, stopwords):
    tokens = text.decode("utf8").split()
    filtered_tokens = [word for word in tokens if word not in stopwords]
    return " ".join(filtered_tokens).encode("utf8")


if __name__ == "__main__":
    corpus=["我 来到 北京 清华大学",
            "他 来到 了 网易 杭研 大厦",
            "小明 硕士 毕业 与 中国 科学院",
            "我 爱 北京 天安门"]
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  #获取词袋模型中的所有词语
    # print word
    # print len(word)
    # weight = tfidf.toarray()  #将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    # print weight.shape
    # for i in range(len(weight)):  #打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    #     print "-------这里输出第", i ,"类文本的词语tf-idf权重------"
    #     for j in range(len(word)):
    #         print word[j].encode("utf8"), weight[i][j]

    # vectorizer = TfidfVectorizer(encoding="utf8")
    # X = vectorizer.fit_transform(corpus)
    # idf = vectorizer.idf_
    # print idf
    # print dict(zip(vectorizer.get_feature_names(), idf))

    # gen_idf("test_tfidft.txt", "idf.txt")


    # dic1 = {'Karl': 1, 'Donald': 1, 'Ifwerson': 1, 'Trump': 0}
    # dic2 = {'Karl': 0, 'Donald': 1, 'Ifwerson': 0, 'Trump': 1}
    # #corpus = ["你 好 吗"]
    # print gen_tf("你 好 吗")

    # stopwords = load_stopwords("../../../../data/processed/stopwords/stopwords-zh.txt")
    # print stopwords
    # text = "中国 你 好 的 吗"
    # print text
    # text2 = remove_stopwords(text, stopwords)
    # print text2
