Please download ppdb-2.0-l-lexical from paraphase.org or ppdb.org (choose `lexical` and `L size`) and put the 800MB file in the root folder.

`python3 baseline.py --pairfile ../data/en-train.txt --valfile ../data/en-val.txt --predfile ../data/pred-en-val.txt --v 1`

`python3 baseline.py --pairfile ../data/en-train.txt --valfile ../data/en-test.txt --predfile ../data/pred-en-test.txt --v 1`

Evaluate things the same way as simple baseline:
`python3 evaluate.py --goldfile ../data/en-val.txt --predfile ../data/pred-en.txt`

Important Note: 

We ran our code for simple baseline (Milestone 2) using a very small test set from one year of the task, i.e. just 250 sentence pairs. When we run our simple baseline on all the sentence pairs available (this is an unsupervised approach and hence no division in train, validation and test set). We achieved an accuracy of 0.3514. 

`python3 evaluate.py --goldfile ../data/en-val.txt --predfile ../data/pred-en-val.txt`

`python3 evaluate.py --goldfile ../data/en-test.txt --predfile ../data/pred-en-test.txt`

Milestone 3 Performance:
Our performance on validation set is 0.451509322045294, while on test set is 0.6579226143671086.
