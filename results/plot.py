import gzip
import cPickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


# unpickle info
filename = 'mnist_model_info.pkl.gz'
f = gzip.open(filename, 'rb')
data = pickle.load(f)
f.close()


# loss
TL = data['training loss']
VL = data['validation loss']
# accuracy
TA = data['training accuracy']
VA = data['validation accuracy']
# error
TE = [(1. - ta) * 100 for ta in TA]
VE = [(1. - va) * 100 for va in VA]

# plot training and validation errors
plt.plot(VE, c='r', lw=1.5)
plt.plot(TE, c='g', lw=1.5)
plt.legend(['validation', 'training'], fontsize=20)
plt.ylim(-1, 10)
plt.xlabel('epoch', size=20)
plt.ylabel('error (%)', size=20)
plt.tight_layout()
plt.savefig('mnist_errors.eps', format='eps', dpi=100)
plt.savefig('mnist_errors.jpg', format='jpg', dpi=100)