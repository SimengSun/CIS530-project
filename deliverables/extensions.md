# Extension-1

## Adding information content using wordnet trees

For this extension, we use wordnet tree representation of two sentences and compute resnik similarity based on least common subsumer (most specific ancestor node). 

We achieve an accuracy of 0.7097 on test set and 0.6226 on validation set.


# Extension-2 

## Adding dense sentence representation using Convolutional Neural Network

![alt text](https://raw.githubusercontent.com/SimengSun/CIS530-project/master/deliverables/pics/cnn-model.png "cnn-model")

For this extension, we combine both baseline features and dense representation of sentence to predict the similarity score of two sentences, such dense representation is a by-product of our model.

Our model is shown as above: the input fed to the CNN is a N-by-2\*D matrix, where N is the maximum length of sentence and D is the dimension of word embedding. Masks are employed to cover a part of the sentence when it is not long enough. Here we set the N to 30 and D to 128. Then we enforce a 1-d convolution operation on this matrix with multiple kernels. Kernel size of 3 and 5 are shown in the figure, however, we use kernel sizes range from 1 to 6 in our actual model. For each kernel size, we randomly initialize 16 kernels, after a non-linear operation(here we use ReLu), max-pooling is used to compute the output of each kernel and finally, we achieve a vector of length 96. This vector encodes in a continuous space the relation of two sentences. To extend our baseline model, we concatenate such representation with some of our baseline features such as overlap_pen feature, absolute difference and mmr_t feature. Simple linear regression model is used to compute the final similarity score. The objective function of our model is to minimize the mean squared error between our predicted value and golden standard score. 

We achieve an accuracy of 0.6460 on test set and 0.6615 on validation set.


# Extension-3

We use InferSent to get the embeddings of all the sentences we have. Given a pair of sentences, if they are semantically similar, the cosine similary between two sentence embeddings are supposed to be high. We extracted the cosine similarity between sentence pairs, added it as a feature, and fed to our Support Vector Regression model.

We achieve an accuracy of 0.7220 on test set and 0.8104 on validation set.
