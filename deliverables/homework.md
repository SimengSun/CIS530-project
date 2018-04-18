### Problem Definition

Your goal in this project is to measure the semantic similarity between a given pair of sentences (what they mean rather than whether they look similar syntactically). 

Semantic Textual Similarity (STS) measures the degree of equivalence in the underlying semantics of paired
snippets of text. While making such an assessment is trivial for humans, constructing algorithms and
computational models that mimic human level performance represents a difficult and deep natural language
understanding (NLU) problem.

Given two sentences, participating systems are asked to return a continuous valued similarity score on a scale from
0 to 5, with 0 indicating that the semantics of the sentences are completely independent and 5 signifying semantic
equivalence. Performance is assessed by computing the Pearson correlation between machine assigned semantic
similarity scores and human judgements.

This problem focuses on determining semantic similarity between monolingual and cross-lingual sentence pairs in
the languages Arabic, English and Spanish. 

#### Example 1:
English: Birdie is washing itself in the water basin.
English Paraphrase: The bird is bathing in the sink.
Spanish: El pa ́jaro se esta ban ̃ando en el lavabo.
Similarity Score: 5 ( The two sentences are completely equivalent, as they mean the same thing.)

#### Example 2:
English: The young lady enjoys listening to the guitar.
English Paraphrase: The woman is playing the violin.
Spanish: La mujer esta ́ tocando el viol ́ın.
Similarity Score: 1 ( The two sentences are not equivalent, but are on the same topic. )

You are free to use any unsupervised or supervised approach for the above mentioned problem. A very simple baseline to start with would be using binary bag-of-words model with the entire vocabulary as features to create embeddings and measuring the cosine similarity between the produced embeddings to generate a final prediction score. You should be able to achieve a Pearson Correlation Coefficient of 0.62 with this very simple approach. 

### Evaluation Metric
Given two sentences, participating systems are asked to return a continuous valued similarity score on a scale from
0 to 5, with 0 indicating that the semantics of the sentences are completely independent and 5 signifying semantic
equivalence. Performance is assessed by computing the Pearson correlation between machine assigned semantic
similarity scores and human judgements.

### Dataset
We will use monolingual \ sentence pairs in English, Spanish and Arabic as training data. There is a
lot of available data from shared tasks in STS from at least the last 5 years. We will use the evaluation data for 2017 shared task as the test set.

### Resources

[STS Shared Task 2017](http://alt.qcri.org/semeval2017/task1/)

[Models submitted to shared task - STS 2017](http://www.aclweb.org/anthology/S17-2001)
