import pprint
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np
import datetime
import pdb

from nltk.corpus import wordnet as wn

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--trainfile', type=str, required=True)
parser.add_argument('--testfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)
parser.add_argument('--v', type=int, required=True)

vpath = "./data/vocab.txt"
V = 10000

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        V = 10002 # 10000 + unk + pad
        D = 256 # word embedding size
        Cin = 1 # input channel
        ks = [1,2,3,4,5,6] # kernel size
        Cout = 32
        dropout = 0.5

        self.embed = nn.Embedding(V, D)
        self.conv = nn.ModuleList([nn.Conv2d(Cin, Cout, (k, 2*D)).double() for k in ks])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(ks)*Cout, 1).float()

    def forward(self, s1, s2):
        # s1: batch_size x maxlen
        x1 = self.embed(s1).double()
        x2 = self.embed(s2).double()
        input = torch.cat([x1, x2], 2) # batch_size x maxlen x 2D
        input = input.unsqueeze(1)  # N x 1 x maxlen x 2D
        out = [F.relu(conv(input).squeeze(3)) for conv in self.conv]  # [(N x Cout x maxlen)] * len(ks)
        out = [F.max_pool1d(z, z.size(2)).squeeze(2) for z in out]  # [(N x Cout)] * len(ks)
        out = torch.cat(out, 1)  # N x len(ks)*Cout
        out = self.dropout(out).float()
        out = self.fc(out).float()
        return out

def load_vocab(vocab, V):
    with open(vocab, 'r') as f:
        word2id, id2word = {}, {}
        cnt = 0
        for line in f.readlines()[:V]:
            pieces = line.split()
            if len(pieces) != 2:
                exit(-1)
            word2id[pieces[0]] = cnt
            id2word[cnt] = pieces[0]
            cnt += 1
    return word2id, id2word

def load_data(fname, w2id):
    """
    :return: data
            list of tuples (s1, s2, score)
            where s1 and s2 are list of index of words in vocab
    """
    def get_indxs(sentence, w2id):
        MAX_LEN = 40
        res = []
        sp = sentence.split()
        for word in sp:
            if word in w2id:
                res.append(w2id[word])
            else:
                res.append(V) # unk
        # pad/cut to MAX_LEN
        if len(res) > MAX_LEN:
            res = res[:MAX_LEN]
        else:
            res += [V+1]*(MAX_LEN-len(res))
        return res

    data = []
    with open(fname, 'r') as f:
        for line in f:
            sp = line.split('\t')
            s1 = get_indxs(sp[0], w2id)
            s2 = get_indxs(sp[1], w2id)
            y = float(sp[2].strip())
            data.append((s1, s2, y))
    return data

def load_example(data):
    random.shuffle(data)
    for i in range(len(data)):
        yield data[i][0], data[i][1], data[i][2]

def mini_batch(data, batch_size):
    gen = load_example(data)
    while True:
        s1_lst, s2_lst, y_lst = [], [], []
        for i in range(batch_size):
            s1, s2, y = next(gen)
            s1_lst.append(s1)
            s2_lst.append(s2)
            y_lst.append(y)
        yield np.array(s1_lst), np.array(s2_lst), np.array(y_lst)

def main(args):

    # load vocab
    w2id, id2w = load_vocab(vpath, V)

    # hyper param
    batch_size = 128
    lr = 0.01

    # load data
    data_train = load_data(args.trainfile, w2id)
    data_test = load_data(args.testfile, w2id)
    cnn = CNN()

    # loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adagrad(cnn.parameters(), lr=lr, weight_decay=0.00005)

    # train
    for epoch in range(10):
        for i, (s1, s2, score) in enumerate(mini_batch(data_train, batch_size)):
            s1 = Variable(torch.from_numpy(s1))
            s2 = Variable(torch.from_numpy(s2))
            score = Variable(torch.from_numpy(score)).float()
            optimizer.zero_grad()
            output = cnn(s1, s2)
            loss = criterion(output, score)
            loss.backward()
            optimizer.step()
            if (i) % 5 == 0:
                print(datetime.datetime.now(), 'Epoch {} batch {} loss: {}' .format(epoch, i, loss.data[0]))

    # evaluate
    cnn.eval()
    res = []
    for item in data_test:
        s1, s2 = np.array([item[0]]), np.array([item[1]])
        s1 = Variable(torch.from_numpy(s1))
        s2 = Variable(torch.from_numpy(s2))
        output = cnn(s1, s2)
        res.append(output.data.cpu().numpy()[0][0])

    # write prediction to file
    with open(args.predfile, 'w') as f:
        for i in res:
            f.write(str(i) + '\n')

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
