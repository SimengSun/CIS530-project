
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

|                 | Validation Set                           | Test Set                         |
|-----------------|----------------|------------|------------|----------|-----------|-----------|
|                 | Pearson        | Ave 5(128) | Ave 0(131) | Pearson  | Ave 5(10) | Ave 0(19) |
| Simple Baseline | 0.428          | 3.274      | 0.532      | 0.633    | 4.088     | 0.623     |
| Gold Standard   | 1              | 5          | 0          | 1        | 5         | 0         |

## Experimental Results

### Published baseline

### Extensions

1. Resnik Similarity using Information Content from Brown Corpus

We used information content generated from the Brown corpus to compute the resnik similarity between paths in the wordnet trees for the given sentence pairs. This approach uses IC of the Least Common Subsumer (most specific ancestor node) to output a score which is used by the Support Vector Regression model. 

We were able to improve upon our model by a slight amount using this extension:
<center>

| model               | Validation Data | Test Data |
| ------------------- |:---------------:|:---------:|
| baseline            | 0.6114          | 0.6989    |
| baseline + InferSent|**0.7220**       |**0.8104** |
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

## Conclusions

## References

[1] Cer et. al, **[SemEval-2017 Task 1: Semantic Textual Similarity
Multilingual and Cross-lingual Focused Evaluation.](https://www.aclweb.org/anthology/S/S17/S17-2001.pdf)** *In Proceedings of the 11th International Workshop on Semantic Evaluations (SemEval-2017)*

[2] Maharjan et. al, **[DT Team at SemEval-2017 Task 1: Semantic Similarity Using Alignments, Sentence-Level Embeddings and Gaussian Mixture Model Output.](http://www.aclweb.org/anthology/S17-2014)** *In Proceedings of the 11th International Workshop on Semantic Evaluations (SemEval-2017)*

[3] Banjade et. al, **[DTSim at SemEval-2016 Task 1: Semantic Similarity Model Including Multi-Level Alignment and Vector-Based Compositional Semantics.](http://www.aclweb.org/anthology/S16-1097)** *In Proceedings of SemEval-2016*

[4] Conneau et. al, **[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](http://aclweb.org/anthology/D17-1070)** *In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)*

## Acknowledgements

We took part in regular meetings with out mentor TA Nitish Gupta who helped us with his thoughts on our ideas and giving us possible directions for our extensions to improve our results. 

## Appendices
