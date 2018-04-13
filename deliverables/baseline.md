Please download ppdb-2.0-l-lexical from paraphase.org or ppdb.org (choose `lexical` and `L size`) and put the 800MB file in the root folder.

`python3 baselines.py --pairfile ../data/en-train.txt --valfile ../data/en-val.txt --predfile ../data/pred-en-val.txt --v 1`

`python3 baselines.py --pairfile ../data/en-train.txt --valfile ../data/en-test.txt --predfile ../data/pred-en-test.txt --v 1`

Evaluate things the same way as simple baseline:

`python3 evaluate.py --goldfile ../data/en-val.txt --predfile ../data/pred-en-val.txt`

`python3 evaluate.py --goldfile ../data/en-test.txt --predfile ../data/pred-en-test.txt`


Our performance on validation set is 0.451509322045294, while on test set is 0.6579226143671086.