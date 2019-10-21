import numpy as np
import cv2
from train import *
from Environment import  *
from Agent import *
from keras.models import load_model

TEST=10



if __name__ == '__main__':
    model=load_model('agent.h5')
    agent=Agent(model)
    trajectory=[]
    matrix = np.asarray([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                         [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


    for i in range(TEST):
        _,_,reward,demo=game(Environment(matrix),agent)
        if reward >= 0:
            trajectory.append(np.asarray(demo))

    for i,j in enumerate(trajectory):
        video_dir = 'output'+str(i+1)+'.avi'
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_writer = cv2.VideoWriter(video_dir, fourcc, 5, (12,12))
        for pic in j:
            video_writer.write(pic.reshape(12,12,1))
        video_writer.release()
    print("finally generate "+str(len(trajectory))+" video")

