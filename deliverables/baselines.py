from collections import defaultdict
import pprint
import argparse
import difflib
from googletrans import Translator
import numpy as np
translator = Translator()
import time
import pdb

from nltk.corpus import wordnet as wn

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--pairfile', type=str, required=True)
parser.add_argument('--valfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)
parser.add_argument('--v', type=int, required=True)

def get_ppdb(fname):
    ppdb = {}
    with open(fname, 'r') as f:
        for line in f.readlines():
            sp = line.split('|||')
            w1, w2 = sp[1], sp[2]
            score = sp[3].split()[0].replace('PPDB2.0Score=', '').lstrip().rstrip()
            ppdb[(w1, w2)] = float(score)
            ppdb[(w2, w1)] = float(score)
    return ppdb

ppdb = get_ppdb('../ppdb-2.0-l-lexical')
print('end loading ppdb-2.0-l')
pdb.set_trace()

#--------------------
def sentence_similarity_simple_baseline(s1, s2):
    def embedding_count(s):
        ret_embedding = defaultdict(int)
        for w in s.split():
            w = w.strip('?.,')
            ret_embedding[w] += 1
        return ret_embedding
    first_sent_embedding = embedding_count(s1)
    second_sent_embedding = embedding_count(s2)
    Embedding1 = []
    Embedding2 = []
    for w in first_sent_embedding:
        Embedding1.append(first_sent_embedding[w])
        Embedding2.append(second_sent_embedding[w])
    ret_score = 0
    if not 0 == sum(Embedding2): 
        #https://stackoverflow.com/questions/6709693/calculating-the-similarity-of-two-lists
        # https://docs.python.org/3/library/difflib.html
        sm= difflib.SequenceMatcher(None,Embedding1,Embedding2)
        ret_score = sm.ratio()*5 
    return ret_score

#--------------------
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn

# =========== util func ==============
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'): return 'n'
    if tag.startswith('V'): return 'v'
    if tag.startswith('J'): return 'a'
    if tag.startswith('R'): return 'r'
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

# =========== feature extraction ==============
def sentence_similarity_word_alignment(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet and ppdb """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2)) 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
    score, count = 0.0, 0
    ppdb_score, align_cnt = 0, 0
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        L = [synset.path_similarity(ss) for ss in synsets2]
        L_prime = L
        L = [l for l in L if l]

        # compute ppdb score
        if len(L) > 0 and max(L) > 0.6:
            align_cnt += 1
            ss2 = synsets2[L_prime.index(max(L))]
            w1, w2 = synset.lemma_names()[0], ss2.lemma_names()[0]
            if (w1, w2) in ppdb:
                ppdb_score += ppdb[(w1, w2)]
            else:
                ppdb_score += 4.5

        # Check that the similarity could have been computed

        if L: 
            best_score = max(L)
            score += best_score
            count += 1
    # Average the values
    if count >0: score /= count
<<<<<<< HEAD

    # ppdb_wa_pen_ua features
    len_s1, len_s2 = len(sentence1), len(sentence2)
    ppdb_score = (1 - 0.4 * (len_s1 + len_s2 - 2*align_cnt) / float(len_s1 + len_s2)) * ppdb_score
    return score, ppdb_score

def extract_overlap_pen(s1, s2):
    """
    :param s1:
    :param s2:
    :return: overlap_pen score
    """
    ss1 = s1.strip().split()
    ss2 = s2.strip().split()
    ovlp_cnt = 0
    for w1 in ss1:
        ovlp_cnt += ss2.count(w1)
    score = 2 * ovlp_cnt / (len(ss1) + len(ss2) + .0)
    return score
=======
    ret = score *5
    return ret
>>>>>>> 9f89828da27c7060542f93e72c1ae47335d215b5

#--------------------
# from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression
import sklearn
def main(args):
    T0 =time.time()
    #----------
    # training 
    first_sents = []
    second_sents = []
    true_score = []
    with open(args.pairfile,'r') as f:
        for line in f.readlines():
            line_split = line.split('\t')
            first_sentence = line_split[0]
            second_sentence = line_split[1]
            gs = line_split[2]
            if 2 == args.v: # translator is a little bit slow
                first_sentence = translator.translate(first_sentence, dest='en').text 
            first_sents.append(first_sentence)
            second_sents.append(second_sentence)
            true_score.append(gs)

    feature_scores = []
    for i in range(len(first_sents)):
        s1 = first_sents[i]
        s2 = second_sents[i]
        scores = [ #sentence_similarity_simple_baseline(s1,s2),
                   *sentence_similarity_word_alignment(s1,s2),
                   extract_overlap_pen(s1, s2)
                   ]
        # cosine similarity
        feature_scores.append(scores)
        if i % 100 == 0:
            print('end ', i)
    pdb.set_trace()
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(feature_scores); X_features = scaler.transform(feature_scores)
    print("Elapsed time:",time.time() - T0,"(preprocessing)")
    clf = LinearRegression(); clf.fit(X_features, true_score)
    #-----------
    # predicting
    first_sents = []
    second_sents = []
    with open(args.valfile,'r') as f_val:
        for line in f_val.readlines():
            line_split = line.split('\t')
            first_sentence = line_split[0]
            second_sentence = line_split[1]
            if 2 == args.v: # translator is a little bit slow
                first_sentence = translator.translate(first_sentence, dest='en').text 
            first_sents.append(first_sentence)
            second_sents.append(second_sentence)

    feature_scores = []
    for i in range(len(first_sents)):
        s1 = first_sents[i]
        s2 = second_sents[i]
        scores = [ #sentence_similarity_simple_baseline(s1,s2),
                   *sentence_similarity_word_alignment(s1,s2),
                   extract_overlap_pen(s1, s2)

                   ]
        # cosine similarity
        feature_scores.append(scores)
        if i % 100 == 0:
            print('end ', i)
    X_features = scaler.transform(feature_scores)
    Y_pred_np = clf.predict(X_features)
    Y_pred_np = [min(5,max(0,p),p) for p in Y_pred_np]
    with open(args.predfile,'w') as f_pred:
        for i in range(len(Y_pred_np)):
            f_pred.write(str(Y_pred_np[i])+'\n')
    print("Elapsed time:",time.time() - T0)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
