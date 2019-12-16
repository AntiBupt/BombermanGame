import numpy as np
from  keras.layers import  Dense,Flatten,Conv2D,MaxPool2D,Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

EPOCH=2
LEARNING_RATE=0.00007
EPSILON=0.97

def get_reward(reward,step,epsilon):
    ans=[reward]
    for _ in range(step-1):
        reward=reward * (1 - epsilon)
        ans.append(reward)
    ans.reverse()
    ans=np.asarray(ans)
    ans -= np.mean(ans)
    ans /= np.std(ans)+0.000001
    return ans

class Agent():
    def __init__(self,model=None):
        if not model:
            self.policy=Sequential()
            self.policy.add(Embedding(101,69,input_length=4))
            self.policy.add(Flatten())
            self.policy.add(Dense(128,activation='tanh'))
            self.policy.add(Dense(5,activation='softmax'))
            self.policy.compile(Adam(lr=LEARNING_RATE),'categorical_crossentropy')
        else:
            self.policy=model
    def action(self,state):
        # return np.argmax(self.policy.predict(state)).reshape(-1)[0]
        position = []
        for i in state:
            p0 = int((i[0]-1) * 10 + i[1] + 1)
            p1 = int((i[2]-1) * 10 + i[3] + 1)
            p2 = int((i[4]-1) * 10 + i[5] + 1)
            p3 = int((i[6]-1) * 10 + i[7] + 1)
            position.append([p0, p1, p2, p3])
        return self.policy.predict(np.asarray(position)).reshape(-1)

    def train(self,trajectory,actions,reward):
        position=[]
        for i in trajectory:
            p0 = int((i[0]-1) * 10 + i[1] + 1)
            p1 = int((i[2]-1) * 10 + i[3] + 1)
            p2 = int((i[4]-1) * 10 + i[5] + 1)
            p3 = int((i[6]-1) * 10 + i[7] + 1)
            position.append([p0,p1,p2,p3])
        reward=get_reward(reward,trajectory.shape[0],EPSILON)
        print("train on " + str(trajectory.shape[0])+" length trajectory")
        batch_size=trajectory.shape[0]
        actions=to_categorical(np.asarray(actions,dtype='int'),num_classes=5)
        actions=actions*reward.reshape(-1,1)
        # print(trajectory.shape,actions.shape)
        self.policy.fit(np.asarray(position),actions,batch_size=batch_size,epochs=EPOCH,verbose=2)

    def bhv_clone(self,trajectory,actions,epoch=2):
        trajectory=trajectory.reshape((-1,12,12,1))
        actions=to_categorical(actions,num_classes=5)
        self.policy.fit(trajectory, actions, batch_size=trajectory.shape[0], epochs=epoch, verbose=2)

    def save(self):
        self.policy.save("agent.h5")