We decide to read following manuscripts according to the review paper 

[Cer, Daniel, et al. "SemEval-2017 Task 1: Semantic Textual Similarity-Multilingual and Cross-lingual Focused Evaluation." arXiv preprint arXiv:1708.00055 (2017).](https://arxiv.org/abs/1708.00055).



1. ECNU - Anant

2. BIT - Simeng

 **[Wu, Hao, et al. "BIT at SemEval-2017 Task 1: Using semantic information space to evaluate semantic textual similarity." Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017). 2017.](http://www.aclweb.org/anthology/S17-2007)**

 This paper introduces three methods to measure semantic textual similarity(STS), all of which rely on jaccard coefficient of information content of  the given sentence pairs. This problem can then be simplified to computing non-overlapping information content(IC) of two sentences. By adding words one by one from each layer of WordNet hierarchy taxonomy and computing information content gain iteratively, their original algorithm is improved since no searching for all subsume concepts is needed. Besides computing the similarity score using single IC feature in an unsupervised way, they also tried to combine sentence alignment and word embedding respectively as extra feature to train supervised models to improve their performance. According to their result, IC combined with word embedding achieves the best result.

3. HCTI - Anant

4. MITRE - Danni

 **[John Henderson, Elizabeth Merkhofer, Laura Strickhart, and
Guido Zarrella. 2017. MITRE at SemEval-2017 Task 1:
Simple semantic similarity. In Proceedings of SemEval2017.](http://www.aclweb.org/anthology/S17-2027)**

 This paper took fourth place overall. The authors created an ensemble of following five systems which each independently predicted a similarity score.
 
 * **TakeLab**: They used LIBSVM to train a support vector regression model with an RBF kernel. Input features were comprised of TakeLab-computed n-gram overlap and word similarity metrics.
 * **Recurrent Convolutional Neural Network**: They extended RNN with four new feature categories. The first was a binary variable that indicates whether both words in the pair were determined to have the same dependency type in their respective parses. They also added three convolutional recurrent neural networks (CRNNs), each of which receive as input a sequence of word embeddings. For each aligned word pair, the first CRNN operates on the embeddings of the aligned words, the second CRNN operates on the squared difference of the embeddings of the aligned words, and the final CRNN operates on the embeddings of the parent words selected by the dependency parse.
 * **Paris: String Similarity**: Paris computes 22 string similarity metrics for a pair of strings, and expanded the number of features to 352. LIBLINEAR used these features to build a L1-regularized logistic regression model.
 * **Simple Alignment Measures**: They built an alignment system on top of word embeddings. L2-regularized logistic regression was used to combine 16 simple features calculated as side-effects of alignments.
 * **Enhanced BiLSTM Inference Model (EBIM)**: They recreated a model which encodes each sentence with a bidirectional LSTM over word2vec embeddings, used a parameter-less attention mechanism to produce a soft alignment matrix for the two sentences, and then did inference over each timestep and its alignment using another LSTM. Two fully-connected layers completed the prediction.

 The semantic similarity estimates of the predictors described above contributed to the final prediction with a weighting determined by L2-regularized logistic regression.


5. FCICU - Danni

 **[Basma Hassan, Samir AbdelRahman, Reem Bahgat, and Ibrahim Farag. 2017. FCICU at SemEval-2017 Task 1: Sense-based language independent semantic textual
similarity approach. In Proceedings of SemEval-2017.](http://www.aclweb.org/anthology/S17-2015)**
 
 
 This paper computes a sense-base alignment using BabelNet and took fifth place overall. BabelNet synsets are multilingual allowing non-English and cross-lingual pairs to be processed similarly to English pairs. Alignment similarity scores are used with two runs: one that combines the scores within a string kernel and another that relied on the proportions of the aligned content words on the two sentences. Both runs average the Babelnet based scores with soft-cardinality. Specifically, the authors set the similarity score by taking the arithmetic mean of: 1) sim<sub>proposed</sub>: a proposed sentence-pair semantic similarity score, and 2) sim<sub>SC</sub>: the surfaced-based similarity function. They pre-processed multilingual input sentences with four different tasks: tokenization, stopwords removal, lemmatization and sense tagging.

6. Compi_LIG - Simeng
- **[Ferrero, Jérémy, et al. "CompiLIG at SemEval-2017 Task 1: Cross-language plagiarism detection methods for semantic textual similarity." arXiv preprint arXiv:1704.01346 (2017).](https://arxiv.org/pdf/1704.01346.pdf)**
- Their system combined syntax-based, dictionary-based, context-based and MT-based methods in both supervised and unsupervised way. For syntactical method, they compute cosine similarity of n-gram representation of two sentences; for dictionary-based method, two sets of words for another language can be obtained from Google Translate, then the summation of  weighted Jaccard distance of such set is used to compute the final score; for context-based method, they use weighted distributed representation of words as sentence embedding and compute cosine similarity, where the weights are computed in dictionary-based method; for MT-based approach, they use monolingual aligner to get aligned utterances and measure a variation of jaccard distance based on inverse document frequency of aligned utterances.


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

