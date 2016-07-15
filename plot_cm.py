import gzip
import cPickle as pickle
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import lasagne as nn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mnist_model import neural_network
from mnist_model import functions
from __helpers__ import load_mnist_data
plt.switch_backend('Agg')


# load model and plot confusion matrix
network = neural_network()
f = gzip.open('./output/mnist_model_weights.pkl.gz', 'rb')
weights = pickle.load(f)
f.close()
nn.layers.set_all_param_values(network, weights)
training_function, validation_function, test_function = functions(network)
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
y_hat = test_function(X_test)
cm = confusion_matrix(y_test, y_hat)
plt.matshow(cm)
plt.title('Confusion matrix\n')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('./output/mnist_confusion_matrix.eps', format='eps')
plt.savefig('./output/mnist_confusion_matrix.jpg', format='jpg')
plt.close()