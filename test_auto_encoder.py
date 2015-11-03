
import numpy as np
import pickle
import gzip

from IPython.display import Image as IPImage
from PIL import Image
from AutoEncoder import AutoEncoder
from FlipBatchIterator import FlipBatchIterator

def get_picture_array(X, index):
    array = X[index].reshape(28,28)
    array = np.clip(array, a_min = 0, a_max = 255)
    return  array.repeat(4, axis = 0).repeat(4, axis = 1).astype(np.uint8())

def get_random_images():
    index = np.random.randint(5000)

    original_image = Image.fromarray(get_picture_array(X, index))
    new_size = (original_image.size[0] * 2, original_image.size[1])
    new_im = Image.new('L', new_size)
    new_im.paste(original_image, (0,0))
    rec_image = Image.fromarray(get_picture_array(X_pred, index))
    new_im.paste(rec_image, (original_image.size[0],0))
    new_im.save('images/orig.png', format="PNG")


print("Loading Autoencoder")
ae = AutoEncoder(
    update_learning_rate = 0.01,
    update_momentum = 0.975,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    regression=True,
    max_epochs= 20
    )
ae.load_params_from("./data/conv_ae.np")

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


print("XShape: ", X.shape)

print("Making Predictions")
X_pred = (ae.predict(X_train)* sigma + mu).reshape(-1, 28, 28).astype(int).clip(0,255).astype('uint8')
print("Pred Shape: ", X_pred.shape)


get_random_images()
IPImage('images/orig.png')



print("Encoding Training Data")
X_encoded=ae.encode(X_train)
print("Encoded Shape: ",X_encoded.shape)

print("Decoding Data")
X_decoded = (ae.decode(X_encoded) * sigma + mu).reshape(-1, 28, 28).astype(int).clip(0,255).astype('uint8')
print("Decoded Shape: ",X_decoded.shape)


# check it worked:
pic_array = get_picture_array(X_decoded, np.random.randint(len(X_decoded)))
image = Image.fromarray(pic_array)
image.save('images/test.png', format="PNG")
IPImage('images/test.png')


print("Done.")


