import pickle
import tensorflow as tf
import numpy as np
# TODO: import Keras layers you need here
from keras.layers import Input, Dense, Flatten, Activation
from keras.models import Sequential, Model

if tf.__version__ > '2.0':
    print("Installed Tensorflow is not 1.x,it is %s" % tf.__version__)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior() 

flags = tf.app.flags # tf.app is removed in v 2.0 , https://www.tensorflow.org/guide/effective_tf2 
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    
    model = Sequential()
    model.add(Flatten(input_shape=(X_train.shape[1:]))) # X_train.shape[1:] is (w,h,d) without number of examples in the set
    model.add(Dense(len(np.unique(y_train)))) # len(np.unique(y_train)) gives back number of classes in dataset
    model.add(Activation('softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TODO: train your model here
    model.fit(X_train,y_train,batch_size=120,epochs=50,shuffle=True,validation_data=(X_val,y_val),verbose=1)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
