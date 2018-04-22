Train on train+val and test on test set:

`python3 baseline.py --pairfile ../data/new_data/all-data-shuffled.txt --valfile ../data/new_data/en-val.txt --predfile ../data/pred-en-val.txt --v 1`

`python3 baseline.py --pairfile ../data/new_data/all-data-shuffled.txt --valfile ../data/new_data/test-data-2017.txt --predfile ../data/pred-en-test.txt --v 1`


Evaluate things the same way as simple baseline.

Important Note: 

We ran our code for simple baseline (Milestone 2) using a very small test set from one year of the task, i.e. just 250 sentence pairs. When we run our simple baseline on all the sentence pairs available (this is an unsupervised approach and hence no division in train, validation and test set). We achieved an accuracy of 0.3514. 

`python3 evaluate.py --goldfile ../data/en-val.txt --predfile ../data/pred-en-val.txt`

`python3 evaluate.py --goldfile ../data/test-data-2017.txt --predfile ../data/pred-en-test.txt`

Milestone 3 Performance:
Our performance on validation set is 0.6704658599497688, while on test set is 0.6644912308518697.
