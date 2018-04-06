import pickle
import os
import pdb
import sys

"""
    pickle file:
    { 
      1: [list of first sentence], 
      2: [list of second sentence], 
     'score': [list of corresponding similarity scores] if a line doesn't have score, use -1 to pad
    }
"""


option = sys.argv[1]
if option == 'to_pickle':
    # en
    en_res = {1:[],2:[],'score':[]}
    path = './data/es-en/'
    dirs = os.listdir(path)
    for dir in dirs:
        if dir.startswith('.'):
            continue
        files = os.listdir(path + dir)
        for file in files:
            if '.gs.' in file or file.startswith('.'):
                continue
            with open(os.path.join(path, dir, file), 'r', errors='replace') as f:
                data = [line.split('\t') for line in f.readlines()]
                sent_1 = [s[0] for s in data]
                sent_2 = [s[1].strip() for s in data]
            with open(os.path.join(path, dir, file.replace('input', 'gs')), 'r', errors='replace') as ff:
                scores = []
                for line in ff:
                    line = line.strip()
                    if '\t' in line:
                        pdb.set_trace()
                    if line == '':
                        scores.append(-1)
                    else:
                        scores.append(float(line))
            assert len(scores) == len(data)
            en_res[1].extend(sent_1)
            en_res[2].extend(sent_2)
            en_res['score'].extend(scores)

    with open('./data/es-en.pkl', 'wb') as f:
        pickle.dump(en_res, f)


if option == 'to_pickle_test':
    # en
    en_res = {1:[],2:[],'score':[]}
    path = './data/es-en-test'
    files = os.listdir(path)
    for file in files:
        if '.gs.' in file or file.startswith('.'):
            continue
        with open(os.path.join(path, file), 'r', errors='replace') as f:
            data = [line.split('\t') for line in f.readlines()]
            sent_1 = [s[0] for s in data]
            sent_2 = [s[1].strip() for s in data]
        with open(os.path.join(path, file.replace('input', 'gs')), 'r', errors='replace') as ff:
            scores = []
            for line in ff:
                line = line.strip()
                if '\t' in line:
                    pdb.set_trace()
                if line == '':
                    scores.append(-1)
                else:
                    scores.append(float(line))
        assert len(scores) == len(data)
        en_res[1].extend(sent_1)
        en_res[2].extend(sent_2)
        en_res['score'].extend(scores)

    with open('./data/es-en-test.pkl', 'wb') as f:
        pickle.dump(en_res, f)

if option == "split_train":
    file = sys.argv[2]
    with open(file, 'rb') as f:
        data = pickle.load(f)
    print('num of pair', len(data[1]))
    res = {'train':{1:[], 2:[], 'score':[]}, 'val':{1:[], 2:[], 'score':[]}}
    res['train'][1] = data[1][:int(len(data[1])*0.8)]
    res['train'][2] = data[2][:int(len(data[1]) * 0.8)]
    res['train']['score'] = data['score'][:int(len(data[1]) * 0.8)]
    res['val'][1] = data[1][int(len(data[1]) * 0.8):]
    res['val'][2] = data[2][int(len(data[1]) * 0.8):]
    res['val']['score'] = data['score'][int(len(data[1]) * 0.8):]

    with open(file.replace('data/', 'data/split-'), 'wb') as f:
        pickle.dump(res, f)