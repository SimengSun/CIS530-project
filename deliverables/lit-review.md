We decide to read following manuscripts according to the review paper 

[Cer, Daniel, et al. "SemEval-2017 Task 1: Semantic Textual Similarity-Multilingual and Cross-lingual Focused Evaluation." arXiv preprint arXiv:1708.00055 (2017).](https://arxiv.org/abs/1708.00055).



1. ECNU - Anant

2. BIT - Simeng

3. HCTI - Anant

4. MITRE - Danni

5. FCICU - Danni

6. Compi_LIG - Simeng

7. LIM_LIG - Yezheng
- **[Ferrero, Jérémy, and Didier Schwab. "LIM-LIG at SemEval-2017 Task1: Enhancing the Semantic Similarity for Arabic Sentences with Vectors Weighting." International Workshop on Semantic Evaluations (SemEval-2017). 2017.](https://hal.archives-ouvertes.fr/hal-01531255/)**
- CBOW model is the basic idea for word embeddings, with some modification. Besides idf weights, this manuscript includes POS weights which is unique to me. 
- From word embeddings to sentence embeddings, they use sum of vectors (which is strange to me).

8. DT_Team - Yezheng
 - **[Maharjan, Nabin, et al. "DT_Team at SemEval-2017 Task 1: Semantic Similarity Using Alignments, Sentence-Level Embeddings and Gaussian Mixture Model Output." Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017). 2017.](http://www.aclweb.org/anthology/S17-2014)**
 - POS-tagging, name-entity recognition as well as  normalization, tokenization, lemmatization are preprocess procedure for word embeddings;
 - From word embeddings to sentence embeddings, this manuscript describes (interesting to me) word alignment; then the similarity score was computed as the sum of the scores for all aligned word-pairs divided by the total length of the given sentence pair. 


9. sent2vec - Yezheng
 - **[Pagliardini, Matteo, Prakhar Gupta, and Martin Jaggi. "Unsupervised learning of sentence embeddings using compositional n-gram features." arXiv preprint arXiv:1703.02507 (2017).](https://arxiv.org/abs/1703.02507)**
 - This is unsupervised learning (not using true similarities in gs file) with similarities of each pair just from cosine similarity of sentence embeddings. The sentence embeddings is sent2vec.

SEF@UHH

