Scoring
-------

The official score is based on weighted Pearson correlation between predicted similarity and human annotated similarity. The higher the score, the better the the similarity prediction result from the algorithm.

`evaluation.py` is the evaluation script. The following script returns the correlation for individual pairs:


       $ python3 evaluation.py --goldfile [gold standard file] -- predfile [prediction file]
       
For example:

       $ python3 evaluate.py --goldfile STS.gs.track5.en-en.txt --predfile STS.pred.track5.en-en.txt

[1] <https://www.aclweb.org/anthology/S/S17/S17-2001.pdf>