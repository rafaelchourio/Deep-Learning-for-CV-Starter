#For avoiding warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(6, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("tanh"))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(16, (5, 5), padding="valid"))
        model.add(Activation("tanh"))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation("tanh"))
        
        model.add(Dense(84))
        model.add(Activation("tanh"))
        
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
        
        