# coding=utf-8
from config import *
import json
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
from graph_tool.all import *
from ccig import *
from sentence_score import *
from sentence_pair_score import *
from bm25 import *


LANGUAGE = "Chinese"
W2V_VOCAB = load_W2V_VOCAB(LANGUAGE)

# build BM25 model
event_file = '../../../../data/raw/event-story-cluster/same_event_doc_pair.txt'
stopwords_file = '../../../../data/processed/stopwords/stopwords-zh.txt'
stopwords = read_stopwords(stopwords_file)
event_datas = read_input(event_file, stopwords)
BM25MODEL, _ = build_bm25(event_datas)


def calc_text_pair_features(text1, text2):
    features = []
    features.append(cal_bm25_sim(BM25MODEL, text1, text2))
    features.append(tfidf_cos_sim(text1, text2, IDF))
    features.append(tf_cos_sim(text1, text2))
    # features.append(lcs(text1, text2))
    # features.append(num_common_words(text1, text2))
    features.append(jaccard_common_words(text1, text2))
    features.append(ochiai_common_words(text1, text2))
    # features.append(Levenshtein_distance(text1, text2))
    # features.append(Levenshtein_ratio(text1, text2))
    # features.append(Levenshtein_jaro(text1, text2))
    # features.append(Levenshtein_jaro_winkler(text1, text2))
    # NOTICE: add other features here
    return features


def assign_graph_node_features(g, sentences1, sentences2, title1=None, title2=None):
    """
    Calculate features for graph nodes, and assign it as node property.
    :param g: aligned concept/sentence interaction graph.
    :param sentences1: left list of sentences.
    :param sentences2: right list of sentences.
    :param title1: left title.
    :param title2: right title.
    :return: graph with node feature property.
    """
    vprop_features_manual = g.new_vertex_property("vector<double>")
    vprop_features_w2v = g.new_vertex_property("vector<string>")

    for v in g.vertices():
        # 1. get left and right concatenated text
        idxs1 = list(set(g.vertex_properties["sentidxs1"][v]))
        idxs2 = list(set(g.vertex_properties["sentidxs2"][v]))
        text1 = [sentences1[i] for i in idxs1]
        text1 = " ".join(text1)
        text2 = [sentences2[i] for i in idxs2]
        text2 = " ".join(text2)
        if g.vertex_properties["name"][v] == TITLE_VERTEX_NAME:
            if title1 is not None:
                text1 = title1
            if title2 is not None:
                text2 = title2

        # 2. get manual features
        v_features = []

        # whether this is a title vertex
        if title1 is not None and title2 is not None:
            if g.vertex_properties["name"][v] == TITLE_VERTEX_NAME:
                v_features.append(1.0)
            else:
                v_features.append(0.0)
        # whether this is an empty vertex
        if g.vertex_properties["name"][v] == EMPTY_VERTEX_NAME:
            v_features.append(1.0)
        else:
            v_features.append(0.0)

        # whether both left and right text exist
        if g.vertex_properties["exist"][v] == "LR":
            v_features.append(1.0)
        else:
            v_features.append(0.0)

        # string similarities
        v_features.extend(calc_text_pair_features(text1, text2))

        vprop_features_manual[v] = v_features

        # 3. get string itself for word2vec based models
        text1 = remove_OOV(text1, W2V_VOCAB)
        text2 = remove_OOV(text2, W2V_VOCAB)
        vprop_features_w2v[v] = [text1, text2]

    g.vertex_properties["features"] = vprop_features_manual
    g.vertex_properties["texts"] = vprop_features_w2v
    return g


def assign_graph_edge_weights(g, sentences1, sentences2):
    """
    Calculate weights for graph edges, and assign it as edge property.
    :param g: aligned concept/sentence interaction graph.
    :param sentences1: left list of sentences.
    :param sentences2: right list of sentences.
    :return: graph with edge weight property.
    """
    # assign scores to edges according to the sentence positions in document pair
    # or according to the text rank scores of sentences in document pair
    eprop_weight_position = g.new_edge_property("double")
    eprop_weight_textrank = g.new_edge_property("double")
    textrank1 = textrank(sentences1)
    textrank2 = textrank(sentences2)

    for e in g.edges():  # here we don't diff "L", "R", "LR" edges
        idxs1 = g.edge_properties["sentidxs1"][e].get_array()
        idxs2 = g.edge_properties["sentidxs2"][e].get_array()
        eprop_weight_position[e] = \
            (sum([score_sentence_position(0, s_idx1, ALPHA, BETA) for s_idx1 in idxs1]) +
             sum([score_sentence_position(0, s_idx2, ALPHA, BETA) for s_idx2 in idxs2])) / 2.0
        eprop_weight_textrank[e] = (sum([textrank1[i] for i in idxs1]) +
                                    sum([textrank2[j] for j in idxs2])) / 2.0

    g.edge_properties["weight_position"] = eprop_weight_position
    g.edge_properties["weight_textrank"] = eprop_weight_textrank
    return g


def assign_graph_label(g, label):
    """
    Assign label to a whole graph.
    :param g: aligned concept/sentence interaction graph.
    :param label: graph label value.
    :return: graph with label.
    """
    g.graph_properties["label"] = g.new_graph_property("int")
    g.graph_properties["label"] = label
    return g


def assign_graph_features(g, category, time1, time2,
                          sentences1, sentences2,
                          title1=None, title2=None):
    """
    Assign global graph features.
    :param g: aligned concept/sentence interaction graph.
    :param category: topic category of the two documents
    :param time1: publish time stamp of doc 1
    :param time2: publish time stamp of doc 2
    :param sentences1: left list of sentences.
    :param sentences2: right list of sentences.
    :param title1: title of doc 1
    :param title2: title of doc 2
    :return: graph with global features.
    """
    g.graph_properties["features"] = g.new_graph_property("vector<double>")
    g.graph_properties["multi_scale_features"] = g.new_graph_property("vector<double>")
    g.graph_properties["all_features"] = g.new_graph_property("vector<double>")

    news_features = []
    news_features.append(abs(time1 - time2) / 86400000)
    news_features.append(category)

    content1 = " ".join(sentences1)
    content2 = " ".join(sentences2)
    sents1_01 = " ".join(sentences1[0:1])
    sents2_01 = " ".join(sentences2[0:1])
    sents1_02 = " ".join(sentences1[0:2])
    sents2_02 = " ".join(sentences2[0:2])
    sents1_03 = " ".join(sentences1[0:3])
    sents2_03 = " ".join(sentences2[0:3])
    sents1_04 = " ".join(sentences1[0:4])
    sents2_04 = " ".join(sentences2[0:4])
    sents1_05 = " ".join(sentences1[0:5])
    sents2_05 = " ".join(sentences2[0:5])
    content_features = calc_text_pair_features(content1, content2)
    sents01_features = calc_text_pair_features(sents1_01, sents2_01)
    sents02_features = calc_text_pair_features(sents1_02, sents2_02)
    sents03_features = calc_text_pair_features(sents1_03, sents2_03)
    sents04_features = calc_text_pair_features(sents1_04, sents2_04)
    sents05_features = calc_text_pair_features(sents1_05, sents2_05)

    g.graph_properties["features"] = content_features
    g.graph_properties["multi_scale_features"] = \
        content_features + sents01_features + sents02_features + \
        sents03_features + sents04_features + sents05_features
    g.graph_properties["all_features"] = \
        content_features + sents01_features + sents02_features + \
        sents03_features + sents04_features + sents05_features + news_features

    g.graph_properties["contains_title"] = g.new_graph_property("bool")
    g.graph_properties["contains_title"] = False
    if title1 is not None and title2 is not None:
        g.graph_properties["title_features"] = g.new_graph_property("vector<double>")
        g.graph_properties["title_features"] = calc_text_pair_features(title1, title2)
        g.graph_properties["contains_title"] = True

    return g


def text_pair2graph_worker(i, df, col_label, col_category,
                           col_time1, col_time2,
                           col_text1, col_text2,
                           col_concepts1, col_concepts2,
                           col_title1=None, col_title2=None,
                           use_cd=True,
                           print_fig=False,
                           betweenness_threshold_coef=1.0, max_c_size=10, min_c_size=3):
    """
    Worker function for parallel.
    Transform a document pair into a CCIG (do community detection) / CIG (no community detection) graph
    :param i: i-th row in pandas data frame
    :param df: the data frame of same event/story dataset
    :param col_XXX: the column name of XXX in df
    :param use_cd: whether perform community detection
    :param print_fig: whether print the fig of constructed graph
    :return: the transformed CCIG/CIG graph from i-th row data
    """
    text1 = df.loc[i][col_text1]
    text2 = df.loc[i][col_text2]

    concepts1 = []
    for col in col_concepts1:
        concepts1.extend(str(df.loc[i][col]).split(","))
    concepts1 = list(set(concepts1))
    concepts2 = []
    for col in col_concepts2:
        concepts2.extend(str(df.loc[i][col]).split(","))
    concepts2 = list(set(concepts2))
    # concepts1 = list(set(remove_stopwords(text1, STOPWORDS).split()))
    # concepts2 = list(set(remove_stopwords(text2, STOPWORDS).split()))

    if col_title1 is not None:
        title1 = df.loc[i][col_title1]
    else:
        title1 = None
    if col_title2 is not None:
        title2 = df.loc[i][col_title2]
    else:
        title2 = None

    label = df.loc[i][col_label]
    category = df.loc[i][col_category]
    time1 = df.loc[i][col_time1]
    time2 = df.loc[i][col_time2]

    g, sentences1, sentences2 = \
        text_pair2ccig(text1, text2, concepts1, concepts2,
                       title1, title2, use_cd,
                       betweenness_threshold_coef, max_c_size, min_c_size)
    if g is None:
        return None

    g = assign_graph_node_features(g, sentences1, sentences2, title1, title2)
    g = assign_graph_edge_weights(g, sentences1, sentences2)
    g = assign_graph_label(g, label)
    g = assign_graph_features(g, category, time1, time2,
                              sentences1, sentences2,
                              title1, title2)
    if print_fig:
        print_ccig(g, sentences1, sentences2)
    print(i)
    return g


def extract_align_graphs_from_docpair_data(infile, col_label, col_category,
                                           col_time1, col_time2,
                                           col_text1, col_text2,
                                           col_concepts1, col_concepts2,
                                           col_title1=None, col_title2=None,
                                           use_cd=True,
                                           parallel=True, extract_range=None,
                                           print_fig=False,
                                           betweenness_threshold_coef=1.0, max_c_size=10, min_c_size=3):
    """
    """
    df = pd.read_csv(infile, sep="|")
    gs = []
    print("parallel extract graph features")
    partial_worker = partial(
        text_pair2graph_worker, df=df, col_label=col_label,
        col_category=col_category,
        col_time1=col_time1, col_time2=col_time2,
        col_text1=col_text1, col_text2=col_text2,
        col_concepts1=col_concepts1, col_concepts2=col_concepts2,
        col_title1=col_title1, col_title2=col_title2,
        use_cd=use_cd,
        print_fig=print_fig,
        betweenness_threshold_coef=betweenness_threshold_coef, max_c_size=max_c_size, min_c_size=min_c_size)
    if extract_range is None:
        extract_range = range(df.shape[0])

    if parallel:
        pool = mp.Pool(processes=mp.cpu_count())
        gs = pool.map(partial_worker, extract_range)
    else:
        for i in extract_range:
            gs.append(partial_worker(i))  # NOTICE: non-parallel can help debug

    return gs


def save_graph_features_to_file(gs, outfile, draw_fig=False):
    """
    Save graphs' topology and node features to a file.
    :param gs: list of graphs.
    :param outfile: output file name that store graphs.
    """
    f = open(outfile, "w")
    i = 0
    for g in gs:
        print("graph " + str(i))
        if g is None:
            print("Graph is None")
            continue
        if draw_fig:
            draw_ccig(g, "ccig_" + str(i) + ".png")

        # used to save graph features
        dict_g = {}

        # graph label
        dict_g["label"] = g.graph_properties["label"]

        # graph features
        dict_g["g_features_vec"] = g.graph_properties["features"].get_array().tolist()
        dict_g["g_multi_scale_features_vec"] = g.graph_properties["multi_scale_features"].get_array().tolist()
        dict_g["g_all_features_vec"] = g.graph_properties["all_features"].get_array().tolist()

        # graph title features
        if g.graph_properties["contains_title"]:
            dict_g["g_title_features_vec"] = g.graph_properties["title_features"].get_array().tolist()
        else:
            dict_g["g_title_features_vec"] = []

        # graph vertices feature and texts matrix
        v_features_mat = []
        v_texts_mat = []
        num_v = g.num_vertices()
        for idx in range(num_v):
            v_features_mat.append(g.vertex_properties["features"][g.vertex(idx)].get_array().tolist())
            v_texts_mat.append(list(g.vertex_properties["texts"][g.vertex(idx)]))  # NOTICE: vector<string> to list, 坑
        dict_g["v_features_mat"] = v_features_mat
        dict_g["v_texts_mat"] = v_texts_mat

        # graph vertices scores matrix
        dict_g["g_vertices_betweenness_vec"] = []
        dict_g["g_vertices_pagerank_vec"] = []
        dict_g["g_vertices_katz_vec"] = []
        for idx in range(num_v):
            dict_g["g_vertices_betweenness_vec"].append(g.vertex_properties["betweenness"][g.vertex(idx)])
            dict_g["g_vertices_pagerank_vec"].append(g.vertex_properties["pagerank"][g.vertex(idx)])
            dict_g["g_vertices_katz_vec"].append(g.vertex_properties["katz"][g.vertex(idx)])

        # graph weighted adjacent matrices
        adj_mat_numsent = np.identity(num_v)
        adj_mat_tfidf = np.identity(num_v)
        adj_mat_position = np.identity(num_v)
        adj_mat_textrank = np.identity(num_v)
        for e in g.edges():
            vsource_idx = g.vertex_index[e.source()]
            vtarget_idx = g.vertex_index[e.target()]

            w_numsent = g.edge_properties["weight_numsent"][e]
            adj_mat_numsent[vsource_idx, vtarget_idx] = w_numsent
            adj_mat_numsent[vtarget_idx, vsource_idx] = w_numsent

            w_tfidf = g.edge_properties["weight_tfidf"][e]
            adj_mat_tfidf[vsource_idx, vtarget_idx] = w_tfidf
            adj_mat_tfidf[vtarget_idx, vsource_idx] = w_tfidf

            w_position = g.edge_properties["weight_position"][e]
            adj_mat_position[vsource_idx, vtarget_idx] = w_position
            adj_mat_position[vtarget_idx, vsource_idx] = w_position

            w_textrank = g.edge_properties["weight_textrank"][e]
            adj_mat_textrank[vsource_idx, vtarget_idx] = w_textrank
            adj_mat_textrank[vtarget_idx, vsource_idx] = w_textrank

        dict_g["adj_mat_numsent"] = adj_mat_numsent.tolist()
        dict_g["adj_mat_tfidf"] = adj_mat_tfidf.tolist()
        dict_g["adj_mat_position"] = adj_mat_position.tolist()
        dict_g["adj_mat_textrank"] = adj_mat_textrank.tolist()

        # write dict to json file
        json_out = json.dumps(dict_g, ensure_ascii=False)  # to dump Chinese
        f.write(json_out)
        f.write("\n")
        i = i + 1
    f.close()


def dataset2featurefile(infile, outfile,
                        col_label, col_category,
                        col_time1, col_time2,
                        col_text1, col_text2,
                        col_concepts1, col_concepts2,
                        col_title1=None, col_title2=None,
                        use_cd=True,
                        parallel=True, extract_range=None,
                        draw_fig=False, print_fig=False,
                        betweenness_threshold_coef=1.0, max_c_size=10, min_c_size=3):
    """
    """
    gs = extract_align_graphs_from_docpair_data(
        infile,
        col_label, col_category,
        col_time1, col_time2,
        col_text1, col_text2,
        col_concepts1, col_concepts2,
        col_title1, col_title2,
        use_cd,
        parallel, extract_range, print_fig,
        betweenness_threshold_coef, max_c_size, min_c_size)
    save_graph_features_to_file(gs, outfile, draw_fig)


if __name__ == "__main__":
    # debug with a few lines
    dataset2featurefile(
        "../../../../data/raw/event-story-cluster/same_event_doc_pair.txt",
        "../../../../data/processed/event-story-cluster/same_event_doc_pair.cd.debug.json",
        "label", "category1", "time1", "time2", "content1", "content2",
        ["keywords1", "ner_keywords1"], ["keywords2", "ner_keywords2"],
        col_title1=None, col_title2=None, use_cd=True,
        draw_fig=True, parallel=False, extract_range=range(2), print_fig=True)

    # process data
    dataset2featurefile(
        "../../../../data/raw/event-story-cluster/same_event_doc_pair.txt",
        "../../../../data/processed/event-story-cluster/same_event_doc_pair.cd.json",
        "label", "category1", "time1", "time2", "content1", "content2",
        ["keywords1", "ner_keywords1"], ["keywords2", "ner_keywords2"],
        col_title1="title1", col_title2="title2", use_cd=True,
        draw_fig=False, parallel=True, extract_range=None,
        betweenness_threshold_coef=1.0, max_c_size=6, min_c_size=2)
    dataset2featurefile(
        "../../../../data/raw/event-story-cluster/same_story_doc_pair.txt",
        "../../../../data/processed/event-story-cluster/same_story_doc_pair.cd.json",
        "label", "category1", "time1", "time2", "content1", "content2",
        ["keywords1", "ner_keywords1"], ["keywords2", "ner_keywords2"],
        col_title1="title1", col_title2="title2", use_cd=True,
        draw_fig=False, parallel=True, extract_range=None,
        betweenness_threshold_coef=1.0, max_c_size=6, min_c_size=2)
    dataset2featurefile(
        "../../../../data/raw/event-story-cluster/same_event_doc_pair.txt",
        "../../../../data/processed/event-story-cluster/same_event_doc_pair.no_cd.json",
        "label", "category1", "time1", "time2", "content1", "content2",
        ["keywords1", "ner_keywords1"], ["keywords2", "ner_keywords2"],
        col_title1="title1", col_title2="title2", use_cd=False,
        draw_fig=False, parallel=True, extract_range=None,
        betweenness_threshold_coef=1.0, max_c_size=6, min_c_size=2)
    dataset2featurefile(
        "../../../../data/raw/event-story-cluster/same_story_doc_pair.txt",
        "../../../../data/processed/event-story-cluster/same_story_doc_pair.no_cd.json",
        "label", "category1", "time1", "time2", "content1", "content2",
        ["keywords1", "ner_keywords1"], ["keywords2", "ner_keywords2"],
        col_title1="title1", col_title2="title2", use_cd=False,
        draw_fig=False, parallel=True, extract_range=None,
        betweenness_threshold_coef=1.0, max_c_size=6, min_c_size=2)
