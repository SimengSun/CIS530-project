Please download ppdb-2.0-l-lexical from paraphase.org or ppdb.org (choose `lexical` and `L size`) and put the 800MB file in the root folder.

`python3 baselines.py --pairfile ../data/en-train.txt --valfile ../data/en-test.txt --valfile ../data/en-val.txt --predfile ../data/pred-en.txt --v 1`

Evaluate things the same way as simple baseline:
`python3 evaluate.py --goldfile ../data/en-val.txt --predfile ../data/pred-en.txt`


Important Note: 

We ran our code for simple baseline (Milestone 2) using a very small test set from one year of the task, i.e. just 250 sentence pairs. When we run our simple baseline on all the sentence pairs available (this is an unsupervised approach and hence no division in train, validation and test set). We achieved an accuracy of 0.3514. 

Our new published baseline achieves a score of 0.44 on the validation set and a score of x.xx on the test set (trained on a combination of both training and validation data)