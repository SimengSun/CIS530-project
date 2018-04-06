
Data
-------

For Spainish-English (es-en), we collect data from 2016, 2017 datasets (there are 103 pairs from 2016, 250 pairs from 2017):<br>
<http://alt.qcri.org/semeval2017/task1/index.php?id=data-and-tools>
<http://ixa2.si.ehu.es/stswiki/index.php/Main_Page>

For English-English (en-en), data is more rich and we collect from 2017, 2015 datasets:
<http://alt.qcri.org/semeval2017/task1/index.php?id=data-and-tools>
<http://ixa2.si.ehu.es/stswiki/index.php/Main_Page><br>
<http://alt.qcri.org/semeval2014/task10/index.php?id=sts-en>


For each group of datasets, it consists of<br>
- a .txt file with pairs of sentences and<br>
- a .txt goldfile with a list of similarity by human judgements; length of list is same as number of pairs in first .txt file.


Score is scaled between 0 and 5 with six levels of similarity summarized in [1].

For monolingual and cross-lingual task, each of them has 3 pickle files.
xx.pkl: includes both training and test data of the format:

            {
                1: [list of first sentence],
                2: [list of second sentence],
                'score': [list of corresponding similarity scores] if a line doesn't
                 have score, use -1 to pad
            }
            
xx-test.pkl: contains all test pairs (evaluation data for 2017 task1)<br>
split-xx.pkl: contains 80%/20% split of training/development data
            
            {
                'train': { SAME FORMAT AS xx.pkl},
                'val':  {SAME FORMAT AS xx.pkl}
            }
           
 [1] <https://www.aclweb.org/anthology/S/S17/S17-2001.pdf>

----------
Run `python3 data/pkl2txt.py` to get reader-friendly TXT file.