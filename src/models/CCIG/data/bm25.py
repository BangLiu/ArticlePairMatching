# coding=utf8
import sys
from gensim.summarization import bm25


def read_stopwords(infile):
    res = {}
    for line in open(infile):
        line = line.strip()
        res[line] = 1
    return res


def read_input(infile, stopwords):
    lineid = 0
    field_num = 0
    datas = []
    docid = 0
    for line in open(infile):
        lineid += 1
        if lineid == 1:
            field_num = len(line.split('|'))
            continue
        line = line.strip()
        fields = line.split('|')
        if len(fields) != field_num:
            sys.stderr.write('format error1\t' + line + '\n')
            continue
        cur_data = {}
        cur_data['label'] = int(fields[0])
        cur_data['doc1'] = {}
        cur_data['doc1']['orig'] = fields[5].strip()
        cur_data['doc1']['text'] = cur_data['doc1']['orig'].replace(' ', '')
        cur_data['doc1']['tokens'] = cur_data['doc1']['orig'].split(' ')
        cur_data['doc1']['tokens_without_stopwords'] = [w for w in cur_data['doc1']['tokens'] if w not in stopwords]
        cur_data['doc1']['docid'] = docid
        docid += 1

        cur_data['doc2'] = {}
        cur_data['doc2']['orig'] = fields[6].strip()
        cur_data['doc2']['text'] = cur_data['doc2']['orig'].replace(' ', '')
        cur_data['doc2']['tokens'] = cur_data['doc2']['orig'].split(' ')
        cur_data['doc2']['tokens_without_stopwords'] = [w for w in cur_data['doc2']['tokens'] if w not in stopwords]
        cur_data['doc2']['docid'] = docid
        docid += 1
        datas.append(cur_data)
    return datas


def get_best_result(sim2labels):
    sorted_sim2labels = sorted(sim2labels, key=lambda x: x[0])
    num1 = sum(x[1] for x in sim2labels)
    num0 = len(sim2labels) - num1
    tn, fn, fp, tp = 0.0, 0.0, float(num0), float(num1)
    best_acc, best_f1, best_threshold = -0.1, -0.1, 0
    for sim2label in sorted_sim2labels:
        sim, label = sim2label[0], sim2label[1]
        if label == 0:
            tn += 1.0
            fp -= 1.0
        else:
            fn += 1.0
            tp -= 1.0
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn)
        if acc > best_acc:
            best_acc, best_f1 = acc, f1
            best_threshold = sim

    return best_threshold, best_acc, best_f1


def build_bm25(datas):
    corpus = []
    docid2index = {}
    for cur_id, cur_data in enumerate(datas):
        corpus.append(cur_data['doc1']['tokens_without_stopwords'])
        docid2index[cur_data['doc1']['docid']] = cur_id * 2
        corpus.append(cur_data['doc2']['tokens_without_stopwords'])
        docid2index[cur_data['doc2']['docid']] = cur_id * 2 + 1
    bm25Model = bm25.BM25(corpus)

    return bm25Model, docid2index


def cal_bm25_sim(bm25Model, tokens1, tokens2):
    ts1, ts2 = tokens1, tokens2
    if len(tokens1) > len(tokens2):
        ts1 = tokens2
        ts2 = tokens1
    freqs = {}
    for word in ts2:
        if word not in freqs:
            freqs[word] = 0
        freqs[word] += 1

    param_k1 = 1.5
    param_b = 0.75
    score1, score2 = 0.0, 0.0
    for word in ts1:
        if word not in freqs or word not in bm25Model.idf:
            continue
        score1 += (bm25Model.idf[word] * freqs[word] * (param_k1 + 1) / (
            freqs[word] + param_k1 * (1 - param_b + param_b * 1)))
    for word in ts2:
        if word not in freqs or word not in bm25Model.idf:
            continue
        score2 += (bm25Model.idf[word] * freqs[word] * (param_k1 + 1) / (
            freqs[word] + param_k1 * (1 - param_b + param_b * 1)))

    sim = score1 / score2 if score2 > 0 else 0
    sim = sim if sim <= 1.0 else 1.0
    return sim


def train(datas, bm25Model, docid2index):
    sim2labels = []
    num0, num1 = 0, 0
    for cur_id, cur_data in enumerate(datas):
        d1 = cur_data['doc1']['tokens_without_stopwords']
        index1 = docid2index[cur_data['doc1']['docid']]
        d2 = cur_data['doc2']['tokens_without_stopwords']
        index2 = docid2index[cur_data['doc2']['docid']]
        if len(d1) < len(d2):
            sim = bm25Model.get_score(d1, index2) / bm25Model.get_score(d2, index2)
        else:
            sim = bm25Model.get_score(d2, index1) / bm25Model.get_score(d1, index1)
        sim = sim if sim <= 1.0 else 1.0
        sim2labels.append((sim, cur_data['label']))
        if cur_data['label'] == 0:
            num0 += 1
        else:
            num1 += 1

    best_threshold, best_acc, best_f1 = get_best_result(sim2labels)
    return best_threshold, best_acc, best_f1


def test(datas, bm25Model, docid2index, best_threshold):
    num0, num1 = 0, 0
    tn, fn, fp, tp = 0.0, 0.0, 0.0, 0.0
    for cur_id, cur_data in enumerate(datas):
        d1 = cur_data['doc1']['tokens_without_stopwords']
        index1 = docid2index[cur_data['doc1']['docid']]
        d2 = cur_data['doc2']['tokens_without_stopwords']
        index2 = docid2index[cur_data['doc2']['docid']]
        if len(d1) < len(d2):
            sim = bm25Model.get_score(d1, index2) / bm25Model.get_score(d2, index2)
        else:
            sim = bm25Model.get_score(d2, index1) / bm25Model.get_score(d1, index1)
        sim = sim if sim <= 1.0 else 1.0
        if sim >= best_threshold:
            pred = 1
            if cur_data['label'] == 0:
                fp += 1.0
                num0 += 1
            else:
                tp += 1.0
                num1 += 1
        else:
            pred = 0
            if cur_data['label'] == 0:
                tn += 1.0
                num0 += 1
            else:
                fn += 1.0
                num1 += 1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)

    return acc, f1


if __name__ == '__main__':
    event_file = '../../../../data/raw/event-story-cluster/same_event_doc_pair.txt'
    stopwords_file = '../../../../data/processed/stopwords/stopwords-zh.txt'

    stopwords = read_stopwords(stopwords_file)
    event_datas = read_input(event_file, stopwords)
    # story_datas = read_input(story_file, stopwords)

    bm25Model, docid2index = build_bm25(event_datas)

    tokens1 = '我们 的 目标 就 是 能够 使用 海量 用户 搜索 日志'
    tokens2 = '在 海量 数据 里 挖掘 潜藏 的 查询 之间 的 结构 信息'
    tokens3 = '我们 的 目标 就 是 能够 使用 海量 用户 搜索 日志'
    for _ in range(5):
        sim = cal_bm25_sim(bm25Model, tokens1, tokens2)
        print(tokens1)
        print(tokens2)
        print(sim)
        print()
        sim = cal_bm25_sim(bm25Model, tokens1, tokens3)
        print(tokens1)
        print(tokens3)
        print(sim)
        print()
