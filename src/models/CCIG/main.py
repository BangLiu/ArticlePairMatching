# coding=utf-8
import time
import argparse
import math
import torch.nn as nn
import torch.optim as optim
from loader import *
from models import *
from models.ema import EMA
from collections import OrderedDict
from util.nlp_utils import *
from util.exp_utils import set_device, set_random_seed, summarize_model


# Training settings
parser = argparse.ArgumentParser()

# cuda, seed
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# data files
parser.add_argument('--inputdata', type=str, default="event-story-cluster/same_story_doc_pair.cd.json", help='input data path')
parser.add_argument('--outputresult', type=str, default="event-story-cluster/same_story_doc_pair.cd.result.txt", help='output file path')
parser.add_argument('--data_type', type=str, default="event", help='event or story')

# train
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--num_data', type=int, default=1000000000, help='maximum number of data samples to use.')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--lr_warm_up_num', default=1000, type=int, help='number of warm-up steps of learning rate')
parser.add_argument('--beta1', default=0.8, type=float, help='beta 1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta 2')
parser.add_argument('--no_grad_clip', default=False, action='store_true', help='whether use gradient clip')
parser.add_argument('--max_grad_norm', default=5.0, type=float, help='global Norm gradient clipping rate')
parser.add_argument('--use_ema', default=False, action='store_true', help='whether use exponential moving average')
parser.add_argument('--ema_decay', default=0.9999, type=float, help='exponential moving average decay')

# model
parser.add_argument('--hidden_vfeat', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout_vfeat', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden_siamese', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--dropout_siamese', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden_final', type=int, default=16, help='Number of hidden units.')

parser.add_argument('--use_gcn', action='store_true', default=False, help='use GCN in model.')
parser.add_argument('--num_gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--use_cd', action='store_true', default=False, help='use community detection in graph.')
parser.add_argument('--use_siamese', action='store_true', default=False, help='use siamese encoding for vertices.')
parser.add_argument('--use_vfeatures', action='store_true', default=False, help='use vertex features.')
parser.add_argument('--use_gfeatures', action='store_true', default=False, help='use global graph features.')

parser.add_argument('--gfeatures_type', type=str, default="features", help='what features to use: features, multi_scale_features, all_features')
parser.add_argument('--gcn_type', type=str, default="valina", help='gcn layer type')
parser.add_argument('--pool_type', type=str, default="mean", help='pooling layer type')
parser.add_argument('--combine_type', type=str, default="separate", help='separate or concatenate, vfeatures with siamese encoding')

# graph
parser.add_argument('--adjacent', type=str, default="tfidf", help='adjacent matrix')
parser.add_argument('--vertice', type=str, default="pagerank", help='vertex centrality')
parser.add_argument('--betweenness_threshold_coef', type=float, default=1.0, help='community detection parameter')
parser.add_argument('--max_c_size', type=int, default=6, help='community detection parameter')
parser.add_argument('--min_c_size', type=int, default=2, help='community detection parameter')

args = parser.parse_args()

# configuration
device, use_cuda, n_gpu = set_device(args.no_cuda)
set_random_seed(args.seed)

# data
print("begin loading W2V............")
LANGUAGE = "Chinese"
W2V = load_w2v("../../../data/raw/word2vec/w2v-zh.model", "Company", 200)
W2V = dict((k, W2V[k]) for k in W2V.keys() if len(W2V[k]) == 200)
W2V = OrderedDict(W2V)
word_to_ix = {word: i for i, word in enumerate(W2V)}
MAX_LEN = 200  # maximum text length for each vertex
embed_size = len(list(W2V.values())[0])
print("W2V loaded! \nVocab size: %d, Embedding size: %d" % (len(word_to_ix), embed_size))

if args.data_type == "event" and args.use_cd:
    args.inputdata = "event-story-cluster/same_event_doc_pair.cd.json"
    args.outputresult = "event-story-cluster/same_event_doc_pair.cd.result.txt"
if args.data_type == "event" and not args.use_cd:
    args.inputdata = "event-story-cluster/same_event_doc_pair.no_cd.json"
    args.outputresult = "event-story-cluster/same_event_doc_pair.no_cd.result.txt"
if args.data_type == "story" and args.use_cd:
    args.inputdata = "event-story-cluster/same_story_doc_pair.cd.json"
    args.outputresult = "event-story-cluster/same_story_doc_pair.cd.result.txt"
if args.data_type == "story" and not args.use_cd:
    args.inputdata = "event-story-cluster/same_story_doc_pair.no_cd.json"
    args.outputresult = "event-story-cluster/same_story_doc_pair.no_cd.result.txt"
print(args)

path = "../../../data/processed/" + args.inputdata
print("begin loading DATA............" + path)
v_texts_w2v_idxs_l_list, v_texts_w2v_idxs_r_list, v_features_list,\
    adjs_numsent_list, adjs_tfidf_list, adjs_position_list, adjs_textrank_list,\
    g_features, g_multi_scale_features, g_all_features,\
    g_vertices_betweenness, g_vertices_pagerank, g_vertices_katz,\
    labels, idx_train, idx_val, idx_test = load_graph_data(path, word_to_ix, MAX_LEN, args.num_data)
print("DATA loaded! \nnumber of samples: %d, max sentence length: %d" % (len(idx_train) + len(idx_val) + len(idx_test), MAX_LEN))

if args.gfeatures_type == "multi_scale_features":
    g_features = g_multi_scale_features
if args.gfeatures_type == "all_features":
    g_features = g_all_features

adjacent_dict = {
    "numsent": adjs_numsent_list,
    "tfidf": adjs_tfidf_list,
    "position": adjs_position_list,
    "textrank": adjs_textrank_list}
g_vertices_dict = {
    "betweenness": g_vertices_betweenness,
    "pagerank": g_vertices_pagerank,
    "katz": g_vertices_katz}
adjs = adjacent_dict[args.adjacent]
g_vertices = g_vertices_dict[args.vertice]

# model
model = SE_GCN(
    args, W2V, MAX_LEN, embed_size,
    nfeat=v_texts_w2v_idxs_l_list[0].shape[1], nfeat_v=v_features_list[0].shape[1], nfeat_g=len(g_features[0]),
    nhid_vfeat=args.hidden_vfeat, nhid_siamese=args.hidden_siamese,
    dropout_vfeat=args.dropout_vfeat, dropout_siamese=args.dropout_siamese,
    nhid_final=args.hidden_final)
summarize_model(model)

if args.use_ema:
    ema = EMA(args.ema_decay)
    ema.register(model)

# optimizer and scheduler
parameters = filter(lambda p: p.requires_grad, model.parameters())
# optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9)
optimizer = optim.Adam(
    params=parameters,
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=1e-8,
    weight_decay=3e-7)
cr = 1.0 / math.log(args.lr_warm_up_num)
scheduler = None
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda ee: cr * math.log(ee + 1)
    if ee < args.lr_warm_up_num else 1)


torch.backends.cudnn.benchmark = True
if use_cuda:
    print("use cuda data")
    model.cuda()
    v_texts_w2v_idxs_l_list = [x.cuda() for x in v_texts_w2v_idxs_l_list]
    v_texts_w2v_idxs_r_list = [x.cuda() for x in v_texts_w2v_idxs_r_list]
    v_features_list = [x.cuda() for x in v_features_list]
    g_features = [x.cuda() for x in g_features]
    adjs = [x.cuda() for x in adjs]
    g_vertices = [x.cuda() for x in g_vertices]
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train_epoch(epoch, step):
    t = time.time()
    t0 = time.time()
    outputs = []
    loss_train = 0.0

    for i in idx_train:
        i = i.cpu().numpy().tolist()
        if i >= args.num_data:
            break
        # get data
        w2v_idxs_l = v_texts_w2v_idxs_l_list[i]
        w2v_idxs_r = v_texts_w2v_idxs_r_list[i]
        v_feature = v_features_list[i]
        g_feature = g_features[i]
        adj = adjs[i]
        g_vertice = g_vertices[i]
        label = labels[i]  # must add []

        # calculate loss and back propagation
        model.train()
        optimizer.zero_grad()
        output = model(w2v_idxs_l, w2v_idxs_r, v_feature, adj, g_feature, g_vertice)  # what if batch > 1 ?
        outputs.append(output.data)
        loss = nn.BCELoss()(output, label)
        loss_train += loss.item()
        loss.backward()

        # gradient clip
        if (not args.no_grad_clip):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # update model
        optimizer.step()

        # update learning rate
        if scheduler is not None:
            scheduler.step()

        # exponential moving avarage
        if args.use_ema:
            ema(model, step)
        step += 1

        # # print training info
        # if i % 1000 == 0:
        #     t1 = time.time()
        #     print("training process......%f%%, time: %fs" % ((i + 0.0) / len(idx_train) * 100, t1 - t0))
        #     t0 = t1

    loss_train = loss_train / len(idx_train)
    acc_train = bc_accuracy(torch.stack(outputs), labels[idx_train.data.cpu().numpy()])
    f1_train = f1score(torch.stack(outputs), labels[idx_train.data.cpu().numpy()])

    # Evaluate validation set performance separately,
    loss_val, acc_val, f1_val, outputs_val = test(
        model, v_texts_w2v_idxs_l_list, v_texts_w2v_idxs_r_list, v_features_list, adjs, g_features, g_vertices, labels, idx_val)

    # print info
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train),
          'acc_train: {:.4f}'.format(acc_train),
          'f1_train: {:.4f}'.format(f1_train),
          'loss_val: {:.4f}'.format(loss_val),
          'acc_val: {:.4f}'.format(acc_val),
          'f1_val: {:.4f}'.format(f1_val),
          'time: {:.4f}s'.format(time.time() - t))
    fout.write('Epoch: ' + str(epoch + 1) + ',' +
               'loss_train: ' + str(loss_train) + ',' +
               'acc_train: ' + str(acc_train) + ',' +
               'f1_train: ' + str(f1_train) + ',' +
               'loss_val: ' + str(loss_val) + ',' +
               "acc_val: " + str(acc_val) + ',' +
               'f1_val: ' + str(f1_val) + '\n')
    return step


def train(args, fout):
    step = 0
    for epoch in range(args.epochs):
        step = train_epoch(epoch, step)
        test_loss, test_acc, test_f1, outputs_test = test(
            model, v_texts_w2v_idxs_l_list, v_texts_w2v_idxs_r_list, v_features_list, adjs, g_features, g_vertices, labels, idx_test)
        print(
            "Test set results:",
            "loss= {:.4f}".format(test_loss),
            "accuracy= {:.4f}".format(test_acc),
            "f1= {:.4f}".format(test_f1))
        fout.write(
            "Test set results:" +
            "loss= " + str(test_loss) + ',' +
            "accuracy= " + str(test_acc) + ',' +
            "f1_score= " + str(test_f1) + '\n')


def test(model, v_texts_w2v_idxs_l_list, v_texts_w2v_idxs_r_list, v_features_list, adjs, g_features, g_vertices, labels, idxs):
    model.eval()
    outputs = []
    output_p = []
    loss_test = 0.0
    for i in idxs:
        w2v_idxs_l = v_texts_w2v_idxs_l_list[i]
        w2v_idxs_r = v_texts_w2v_idxs_r_list[i]
        v_feature = v_features_list[i]
        g_feature = g_features[i]
        adj = adjs[i]
        g_vertice = g_vertices[i]
        label = labels[i]
        output = model(w2v_idxs_l, w2v_idxs_r, v_feature, adj, g_feature, g_vertice)
        output_p.append(output.data.cpu().numpy()[0])
        outputs.append(output.data.cpu())
        loss = nn.BCELoss()(output, label)
        loss_test += loss.data.item()
    loss = loss_test / len(idxs)
    acc = bc_accuracy(torch.stack(outputs), labels[idxs.data.cpu().numpy()])
    f1 = f1score(torch.stack(outputs), labels[idxs.data.cpu().numpy()])
    return loss, acc, f1, output_p


def write_to_file(fin, label_list, pred_list):
    with open(fin, 'w') as f:
        for i in range(len(label_list)):
            f.write(str(label_list[i]) + '\t' + str(pred_list[i]) + '\n')


if __name__ == '__main__':
    t_total = time.time()
    outpath = "../../../data/result/" + args.outputresult
    fout = open(outpath, 'w')
    train(args, fout)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    fout.close()
