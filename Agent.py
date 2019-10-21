import numpy as np
from  keras.layers import  Dense,Flatten,Conv2D,MaxPool2D
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.utils import to_categorical

EPOCH=2
LEARNING_RATE=0.001

class Agent():
    def __init__(self,model=None):
        if not model:
            self.policy=Sequential()
            self.policy.add(Conv2D(12,kernel_size=(3,3),padding='valid',input_shape=(12,12,1)))
            self.policy.add(Flatten())
            self.policy.add(Dense(72, activation='sigmoid'))
            self.policy.add(Dense(20, activation='sigmoid'))
            self.policy.add(Dense(5,activation='softmax'))
            self.policy.compile(Nadam(lr=LEARNING_RATE),'categorical_crossentropy')
        else:
            self.policy=model
    def action(self,state):
        return np.argmax(self.policy.predict(state)).reshape(-1)[0]

    def train(self,trajectory,actions,reward):
        trajectory=np.asarray(trajectory).reshape((-1,12,12,1))
        print("train on " + str(trajectory.shape[0])+" lenght trajectory")
        batch_size=trajectory.shape[0]
        actions=to_categorical(np.asarray(actions,dtype='int'),num_classes=5)
        actions=actions*reward
        # print(trajectory.shape,actions.shape)
        self.policy.fit(trajectory,actions,batch_size=batch_size,epochs=EPOCH,verbose=2)

    def bhv_clone(self,trajectory,actions,epoch=2):
        trajectory=trajectory.reshape((-1,12,12,1))
        actions=to_categorical(actions,num_classes=5)
        self.policy.fit(trajectory, actions, batch_size=trajectory.shape[0], epochs=epoch, verbose=2)

    def save(self):
        self.policy.save("agent.h5")