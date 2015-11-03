from lasagne import layers
from nolearn.lasagne import NeuralNet
import numpy as np


class AutoEncoder(NeuralNet):
    def __init__(self, **kwargs):
        conv_filters = 10
        deconv_filters = 10
        filter_sizes = 7
        encode_size = 40
        img_size =28
        super(AutoEncoder,self).__init__(layers=[
                    (layers.InputLayer    , dict(name ='input',
                                                 shape=(None,1,img_size,img_size))),

                    (layers.Conv2DLayer   , dict(name='conv',
                                                 num_filters=conv_filters,
                                                 filter_size = (filter_sizes, filter_sizes),
                                                 pad='valid',
                                                 nonlinearity=None)),

                    (layers.Pool2DLayer, dict(name='pool',
                                              pool_size=(2, 2),
                                              mode ='max')),

                    (layers.DenseLayer    , dict(name='encode_layer',
                                                 num_units = encode_size)),

                    (layers.DenseLayer    , dict(name='hidden',
                                                 num_units= deconv_filters*(img_size+filter_sizes-1)**2/4)),  # output_dense

                    (layers.ReshapeLayer  , dict(name='unflatten',
                                                 shape=([0],deconv_filters,(img_size+filter_sizes-1)//2,(img_size+filter_sizes-1)//2 ))),

                    (layers.Upscale2DLayer, dict(name='unpool',
                                                 scale_factor=(2, 2))),

                    (layers.Conv2DLayer   , dict(name='deconv',
                                                 num_filters=1,
                                                 filter_size = (filter_sizes, filter_sizes),
                                                 pad="valid",
                                                 nonlinearity=None)),

                    (layers.ReshapeLayer  , dict(name='output_layer',
                                                 shape = ([0], -1)))
                    ],
            **kwargs
        )
        self.initialize()
        self.encode_idx = [el.name for el in self.get_all_layers()].index('encode_layer')

    @staticmethod
    def get_output_from_layer(layer,X):
        #Perform the actual Decode
        indices = np.arange(128, X.shape[0], 128)

        # not splitting into batches can cause a memory error
        X_batches = np.split(X, indices)
        out = []
        for count, X_batch in enumerate(X_batches):
            res = layers.get_output(layer,X_batch).eval()
            out.append(res)
        return np.vstack(out)

    def encode(self, X):
        encode_layer = self.get_all_layers()[self.encode_idx]
        return self.get_output_from_layer(encode_layer,X)

    def decode(self, X):
        #Get the needed Layers for the Decoder
        encode_layer=self.get_all_layers()[self.encode_idx]
        next_layer = self.get_all_layers()[self.encode_idx + 1]
        final_layer = self.get_all_layers()[-1]

        #Create a new layer for the decoder input
        new_layer = layers.InputLayer(shape = (None, encode_layer.num_units))
        next_layer.input_layer = new_layer

        #Perform the actual Decode
        result = self.get_output_from_layer(final_layer,X)

        #Reconnect he Layers to original state
        next_layer.input_layer=encode_layer

        return result


# # Neural Network with 167921 learnable parameters
#
# ## Layer information
#
#   #  name          size
# ---  ------------  --------
#   0  input         1x28x28
#   1  conv          10x22x22
#   2  pool          10x11x11
#   3  encode_layer  40
#   4  hidden        2890.0
#   5  unflatten     10x17x17
#   6  unpool        10x34x34
#   7  deconv        1x28x28
#   8  output_layer  784
#
#   epoch    train loss    valid loss    train/val  dur
# -------  ------------  ------------  -----------  ------
#       1       0.46078       0.25438      1.81135  93.46s
#       2       0.20494       0.17439      1.17523  93.78s
#       3       0.16487       0.15459      1.06652  93.78s
#       4       0.15123       0.14568      1.03811  94.28s
#       5       0.14347       0.14054      1.02085  94.18s