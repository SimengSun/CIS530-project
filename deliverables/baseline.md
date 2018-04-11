Please download ppdb-2.0-l-lexical from paraphase.org or ppdb.org (choose `lexical` and `L size`) and put the 800MB file in the root folder.

`python3 baselines.py --pairfile ../data/en-train.txt --valfile ../data/en-test.txt --valfile ../data/en-val.txt --predfile ../data/pred-en.txt --v 1`

Evaluate things the same way as simple baseline:
`python3 evaluate.py --goldfile ../data/en-train.txt --predfile ../data/pred_en.txt`