# Semantic textual similarity
## Milestone 2
### Data 
See [data/data.md](data/data.md) for our specification while [data/](data/) contains reader-friendly `.txt` files.
### Evaluation 
See [deliverables/scoring.md](deliverables/scoring.md). To be concise, there are three evluation criteria:
1. Pearson correlation suggested by [^fn1].
2. ranking criteria (of two sequences of similarities);
3. double check of similarities for pairs with underlying 5.0, 0.0 similarities.

Unfortunately, we have not found references for the latter two more convincing/ intepretable evaluation criteria.
### Simple baseline
See [deliverables/simple-baseline.md](deliverables/simple-baseline.md).
## Milestone 3
### Literature review
According to  [^fn1], we compile a summary of literature review in [deliverables/lit-review.md](deliverables/lit-review.md). At least, two interpretable questions are focused: 

1. whether or not the method utilizes gold-standard similarities;
2. choice of sentence embeddings.
### Published basline
See [deliverables/baseline.md](deliverables/baseline.md) for our published baseline.


[^fn1] [Cer, Daniel, et al. "SemEval-2017 Task 1: Semantic Textual Similarity-Multilingual and Cross-lingual Focused Evaluation." arXiv preprint arXiv:1708.00055 (2017).](https://arxiv.org/abs/1708.00055).
