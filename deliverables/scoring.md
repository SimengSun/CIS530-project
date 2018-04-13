Scoring
-------

The official score is based on weighted Pearson correlation between predicted similarity and human annotated similarity. The higher the score, the better the the similarity prediction result from the algorithm.

`evaluation.py` is the evaluation script. The following script returns the correlation for individual pairs:


       $ python3 evaluate.py --goldfile [gold standard file] -- predfile [prediction file]
       
For example:

       $ python3 evaluate.py --goldfile ../data/en-val.txt --predfile ../data/pred-en.txt


However, since we do not see Pearson correlation interpretable, we propose two additional criteria:
1. ranking criteria (of two sequences of similarities);
2. double check of similarities for pairs with underlying 5.0, 0.0 similarities.



[1] <https://www.aclweb.org/anthology/S/S17/S17-2001.pdf>