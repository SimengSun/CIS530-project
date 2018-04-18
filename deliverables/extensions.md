
# Extension-1 

## Adding Dense sentence representation using Convolutional Neural Network

![alt text](https://raw.githubusercontent.com/SimengSun/CIS530-project/master/deliverables/pics/cnn-model.png "cnn-model")

For this extension, we combine both baseline features and dense representation of sentence to predict the similarity score of two sentences, such dense representation is a by-product of our model.

Our model is shown as above: the input fed to the CNN is a N-by-2\*D matrix, where N is the maximum length of sentence and D is the dimension of word embedding. Masks are employed to cover a part of the sentence when it is not long enough. Here we set the N to 30 and D to 128. Then we enforce a 1-d convolution operation on this matrix with multiple kernels. Kernel size of 3 and 5 are shown in the figure, however, we use kernel sizes range from 1 to 6 in our actual model. For each kernel size, we randomly initialize 16 kernels, after a non-linear operation(here we use ReLu), max-pooling is used to compute the output of each kernel and finally, we achieve a vector of length 96. This vector encodes in a continuous space the relation of two sentences. To extend our baseline model, we concatenate such representation with some of our baseline features such as overlap_pen feature, absolute difference and mmr_t feature. Simple linear regression model is used to compute the final similarity score. The objective function of our model is to minimize the mean squared error between our predicted value and golden standard score. 

This model outperforms our baseline model

| model        | Correlation           |
| ------------- |:-------------:|
| baseline      | 0.6502 |
| extension-CNN |  0.7381    |



# Extension-2

# Extension-3

# Extension-4
