
# Semantic Textual Similarity

#### Yezheng Li&emsp;&emsp;&emsp;&emsp;&emsp;Danni Ma&emsp;&emsp;&emsp;&emsp;&emsp;Anant Maheshwari&emsp;&emsp;&emsp;&emsp;&emsp;Simeng Sun


## Abstract
Semantic Textual Similarity (STS) measures the meaning similarity of sentences. Applications of this task include machine translation, summarization, text generation, question answering, short answer grading, semantic search, dialogue and conversational systems. We developed Support Vector Regression model with various features including the similarity scores calculated using alignment-based methods and semantic composition based methods. We have also trained sentence semantic representations with BiLSTM and Convolutional Neural Networks (CNN). The correlations between our system output the human ratings were above 0.8 in the test dataset.

## Introduction
The goal of this task is to measure semantic textual similarity between a given pair of sentences (what they mean rather than whether they look similar syntactically). While making such an assessment is trivial for humans, constructing algorithms and computational models that mimic human level performance represents a difficult and deep natural language
understanding (NLU) problem.

#### Example 1:

English: Birdie is washing itself in the water basin.

English Paraphrase: The bird is bathing in the sink.

Similarity Score: 5 ( The two sentences are completely equivalent, as they mean the same thing.)

#### Example 2:

English: The young lady enjoys listening to the guitar.

English Paraphrase: The woman is playing the violin.

Similarity Score: 1 ( The two sentences are not equivalent, but are on the same topic. )

Semantic Textual Similarity (STS) measures the degree of equivalence in the underlying semantics of paired snippets of text. STS differs from both textual entailment and paraphrase detection in that it captures gradations of meaning overlap rather than making binary classifications of particular relationships. While semantic relatedness expresses a graded semantic relationship as well, it is non-specific about the nature of the relationship with contradictory material still being a candidate for a high score (e.g., “night” and “day” are highly
related but not particularly similar). The task involves producing real-valued similarity scores for sentence pairs. Performance is measured by the Pearson correlation of machine scores with human judgments.

STS is an annual shared task in SemEval since 2012. The STS shared tast data sets have been used extensively for research on sentence level similarity and semantic representations. We have access to STS benchmark which is a new shared training and evaluation set carefully selected from the corpus of English STS shared task data (2012-2017). Over the past five years, numerous participating teams, diverse approaches, and ongoing improvements to state-of-the-art methods have constantly raised the standard of this task.

## Literature Review

**** Anant Polish this ****

[Cer, Daniel, et al. "SemEval-2017 Task 1: Semantic Textual Similarity-Multilingual and Cross-lingual Focused Evaluation." arXiv preprint arXiv:1708.00055 (2017).](https://arxiv.org/abs/1708.00055).


1. ECNU

**[Junfeng Tian, Zhiheng Zhou, Man Lan, and Yuanbin Wu. 2017. ECNU at SemEval-2017 Task 1: Leverage kernel- based traditional nlp features and neural networks to build a universal model for multilingual and cross-lingual seman- tic textual similarity. In Proceedings of SemEval-2017.](http://www.aclweb.org/anthology/S17-2028)**

This model is an average ensemble of three feature engineered models and four deep learning models. Three feature engineered models use Random Forest, Gradient Boosting and XGBoost regression methods which include approaches like edit distance and tree kernels. In neural network models, each network feeds the network feeds the element-wise multiplication, subtraction and concatenation of paired sentence embeddings.

2. BIT

 **[Wu, Hao, et al. "BIT at SemEval-2017 Task 1: Using semantic information space to evaluate semantic textual similarity." Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017). 2017.](http://www.aclweb.org/anthology/S17-2007)**

 This paper introduces three methods to measure semantic textual similarity(STS), all of which rely on jaccard coefficient of information content of  the given sentence pairs. This problem can then be simplified to computing non-overlapping information content(IC) of two sentences. By adding words one by one from each layer of WordNet hierarchy taxonomy and computing information content gain iteratively, their original algorithm is improved since no searching for all subsume concepts is needed. Besides computing the similarity score using single IC feature in an unsupervised way, they also tried to combine sentence alignment and word embedding respectively as extra feature to train supervised models to improve their performance. According to their result, IC combined with word embedding achieves the best result.

3. MITRE

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

4. FCICU

 **[Basma Hassan, Samir AbdelRahman, Reem Bahgat, and Ibrahim Farag. 2017. FCICU at SemEval-2017 Task 1: Sense-based language independent semantic textual
similarity approach. In Proceedings of SemEval-2017.](http://www.aclweb.org/anthology/S17-2015)**
 
 
 This paper computes a sense-base alignment using BabelNet and took fifth place overall. BabelNet synsets are multilingual allowing non-English and cross-lingual pairs to be processed similarly to English pairs. Alignment similarity scores are used with two runs: one that combines the scores within a string kernel and another that relied on the proportions of the aligned content words on the two sentences. Both runs average the Babelnet based scores with soft-cardinality. Specifically, the authors set the similarity score by taking the arithmetic mean of: 1) sim<sub>proposed</sub>: a proposed sentence-pair semantic similarity score, and 2) sim<sub>SC</sub>: the surfaced-based similarity function. They pre-processed multilingual input sentences with four different tasks: tokenization, stopwords removal, lemmatization and sense tagging.

5. Compi_LIG
- **[Ferrero, Jérémy, et al. "CompiLIG at SemEval-2017 Task 1: Cross-language plagiarism detection methods for semantic textual similarity." arXiv preprint arXiv:1704.01346 (2017).](https://arxiv.org/pdf/1704.01346.pdf)**
- Their system combined syntax-based, dictionary-based, context-based and MT-based methods in both supervised and unsupervised way. For syntactical method, they compute cosine similarity of n-gram representation of two sentences; for dictionary-based method, two sets of words for another language can be obtained from Google Translate, then the summation of  weighted Jaccard distance of such set is used to compute the final score; for context-based method, they use weighted distributed representation of words as sentence embedding and compute cosine similarity, where the weights are computed in dictionary-based method; for MT-based approach, they use monolingual aligner to get aligned utterances and measure a variation of jaccard distance based on inverse document frequency of aligned utterances.


6. LIM_LIG
- **[Ferrero, Jérémy, and Didier Schwab. "LIM-LIG at SemEval-2017 Task1: Enhancing the Semantic Similarity for Arabic Sentences with Vectors Weighting." International Workshop on Semantic Evaluations (SemEval-2017). 2017.](https://hal.archives-ouvertes.fr/hal-01531255/)**
- CBOW model is the basic idea for word embeddings, with some modification. Besides idf weights, this manuscript includes POS weights which is unique to me. 
- From word embeddings to sentence embeddings, they use sum of vectors (which is strange to me).

7. DT_Team
 - **[Maharjan, Nabin, et al. "DT_Team at SemEval-2017 Task 1: Semantic Similarity Using Alignments, Sentence-Level Embeddings and Gaussian Mixture Model Output." Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017). 2017.](http://www.aclweb.org/anthology/S17-2014)**
 - unsupervised and supervised learning;
 - POS-tagging, name-entity recognition as well as  normalization, tokenization, lemmatization are preprocess procedure for word embeddings;
 - From word embeddings to sentence embeddings, this manuscript describes (interesting to me) word alignment; then the similarity score was computed as the sum of the scores for all aligned word-pairs divided by the total length of the given sentence pair.


8. sent2vec
 - **[Pagliardini, Matteo, Prakhar Gupta, and Martin Jaggi. "Unsupervised learning of sentence embeddings using compositional n-gram features." arXiv preprint arXiv:1703.02507 (2017).](https://arxiv.org/abs/1703.02507)**
 - This is unsupervised learning (not using true similarities in gs file) with similarities of each pair just from cosine similarity of sentence embeddings. The sentence embeddings is sent2vec.

## Experimental Design

### Data
We obtained the data by merging data from year 2012 to 2017 SemEval Shared Task. Out of a total of approximately 28000 sentence pairs, we were left with about 15115 sentence pairs (cleaning involved removing sentence pairs without a tab delimiter and pairs with a blank gold score). We split the data into three parts as below. 

Training data: 13365 pairs
Validation data:  1500 pairs
Test data: 250 pairs (Same as used by the other teams to test their model in the 2017 task)

#### Data Pre-processing

We used tokenization and lemmatization on the data as a pre-processing step before we turn the data into our models. We chose to do this step as lemmatization does not take away any semantic information from sentences and hence was an essential step for our application.

### Evaluation Metric
The official score is based on weighted Pearson correlation between predicted similarity and human annotated similarity. The higher the score, the better the the similarity prediction result from the algorithm.

### Simple baseline

For the simple baseline, we used an unsupervised approach by creating sentence vectors with each dimension representing whether an individual word appears in a sentence. The final score is calculated using cosine similairty between the sentence vectors. 

We achieved the following results using the simple baseline: 

|                 | Validation Set |            |            | Test Set |           |           |
|-----------------|----------------|------------|------------|----------|-----------|-----------|
|                 | Pearson        | Ave 5(128) | Ave 0(131) | Pearson  | Ave 5(10) | Ave 0(19) |
| Simple Baseline | 0.428          | 3.274      | 0.532      | 0.633    | 4.088     | 0.623     |
| Gold Standard   | 1              | 5          | 0          | 1        | 5         | 0         |

## Experimental Results

### Published baseline

**** Danni Write This ****

### Extensions

1. Resnik Similarity using Information Content from Brown Corpus

We used information content generated from the Brown corpus to compute the resnik similarity between paths in the wordnet trees for the given sentence pairs. This approach uses IC of the Least Common Subsumer (most specific ancestor node) to output a score which is used by the Support Vector Regression model. 

We were able to improve upon our model by a slight amount using this extension:
<center>

| model               | Validation Data | Test Data |
| ------------------- |:---------------:|:---------:|
| baseline            | 0.6114          | 0.6989    |
| baseline + IC       | 0.6226          | 0.7097    |
Table: Pearson Correlations between system outputs and human ratings on different models
</center>

2. Convolution Neural Networks to generate sentence embeddings

**** Simeng write here ****

<center>

| model               | Validation Data | Test Data |
| ------------------- |:---------------:|:---------:|
| baseline            | 0.6114          | 0.6989    |
| baseline + CNN      | 0.6615          | 0.6460    |
Table: Pearson Correlations between system outputs and human ratings on different models
</center>

3. Use InferSent trained sentence embeddings

[InferSent](https://research.fb.com/downloads/infersent/) is a sentence embeddings method that provides semantic sentence representations. It is trained on natural language inference data and generalizes well to many different tasks.

We use InferSent to get the embeddings of all the sentences we have. Given a pair of sentences, if they are semantically similar, the cosine similary between two sentence embeddings are supposed to be high. We extracted the cosine similarity between sentence pairs, added it as a feature, and fed to our Support Vector Regression model.

With the help of InferSent trained sentence representations, the model outperforms baseline model on both validation data and test data:
<center>

| model               | Validation Data | Test Data |
| ------------------- |:---------------:|:---------:|
| baseline            | 0.6114          | 0.6989    |
| baseline + InferSent|**0.7220**       |**0.8104** |
Table: Pearson Correlations between system outputs and human ratings on different models
</center>

### Error Analysis

**** Yezheng Write Here ****

## Conclusions

**** Anant Write This **** 

## References

[1] Cer et. al, **[SemEval-2017 Task 1: Semantic Textual Similarity
Multilingual and Cross-lingual Focused Evaluation.](https://www.aclweb.org/anthology/S/S17/S17-2001.pdf)** *In Proceedings of the 11th International Workshop on Semantic Evaluations (SemEval-2017)*

[2] Maharjan et. al, **[DT Team at SemEval-2017 Task 1: Semantic Similarity Using Alignments, Sentence-Level Embeddings and Gaussian Mixture Model Output.](http://www.aclweb.org/anthology/S17-2014)** *In Proceedings of the 11th International Workshop on Semantic Evaluations (SemEval-2017)*

[3] Banjade et. al, **[DTSim at SemEval-2016 Task 1: Semantic Similarity Model Including Multi-Level Alignment and Vector-Based Compositional Semantics.](http://www.aclweb.org/anthology/S16-1097)** *In Proceedings of SemEval-2016*

[4] Conneau et. al, **[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](http://aclweb.org/anthology/D17-1070)** *In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)*

## Acknowledgements

We took part in regular meetings with out mentor TA Nitish Gupta who helped us with his thoughts on our ideas and giving us possible directions for our extensions to improve our results. 

## Appendices
