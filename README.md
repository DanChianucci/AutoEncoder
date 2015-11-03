# AutoEncoder
An Autoencoder Class inspired by github.com/mikesj-public/convolutional_autoencoder

Rather than creating a nolearn.NeuralNetwork with the specified layers, this NeuralNetwork sub class has the layers built in.

The main reason for this change was to make instantiating the network easier, as well as allowing for an encode and decode function which didn't destroy the original trained network in the process.

I had to install nolearn, theano, and lasagne using the instructions provided at https://pythonhosted.org/nolearn/
