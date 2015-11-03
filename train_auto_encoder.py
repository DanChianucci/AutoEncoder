from FlipBatchIterator import FlipBatchIterator
from AutoEncoder import AutoEncoder

import numpy as np
import pickle
import gzip

print("Unpickling MNIST")
fname = './data/mnist.pkl.gz'
f = gzip.open(fname, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
f.close()

print("Partitioning Data")
X, y = train_set
X = np.rint(X * 256).astype(np.int).reshape((-1, 1, 28, 28))  # convert to (0,255) int range (we'll do our own scaling)
mu, sigma = np.mean(X.flatten()), np.std(X.flatten())
X_train = X.astype(np.float64)
X_train = (X_train - mu) / sigma
X_train = X_train.astype(np.float32)
X_out = X_train.reshape((X_train.shape[0], -1))


print("Begin Training")
epochs = 20
ae = AutoEncoder(
    update_learning_rate = 0.01,
    update_momentum = 0.975,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    regression=True,
    max_epochs= epochs,
    verbose=1,
    )
ae.fit(X_train, X_out,5)

print("Saving Parameters")
ae.save_params_to("./data/conv_ae.np")


print("Done")
