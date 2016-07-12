# deep-learning-mnist

### Description:

MNIST Digit classification using Python (Lasagne + Theano library). Training takes about 20 minutes. All output, training information and weights are saved in the 'results' folder.

Run/theano settings: ```THEANO_FLAGS='mode=FAST_RUN, device=gpu, floatX=float32' python mnist_model.py```

### Results:

* 98.71% accuracy on training data
* 99.13% accuracy on validation data
* 99.36% accuracy on test data.

<img src="./results/mnist_errors.jpg">

### References:

[1] Dataset: [deeplearning.net/data/mnist/](deeplearning.net/data/mnist/)

[2] Lasagne documentation: [lasagne.readthedocs.io/](lasagne.readthedocs.io/)

[3] Lasagne examples: [github.com/Lasagne/Recipes](github.com/Lasagne/Recipes)

[4] Theano documentation: [deeplearning.net/software/theano/](deeplearning.net/software/theano/)