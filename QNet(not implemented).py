import numpy as np
from  keras.layers import  Dense,Flatten,Conv2D,MaxPool2D
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.utils import to_categorical

EPOCH=2
LEARNING_RATE=0.001

class Qnet():
    def __init__(self,model=None):
        if not model:
            self.policy=Sequential()
            self.policy.add(Conv2D(12,kernel_size=(3,3),padding='valid',input_shape=(12,12,1)))
            self.policy.add(Flatten())
            self.policy.add(Dense(72, activation='relu'))
            self.policy.add(Dense(20, activation='relu'))
            self.policy.add(Dense(1))
            self.policy.compile(Nadam(lr=LEARNING_RATE),'mse')

    def train(self):
        pass
