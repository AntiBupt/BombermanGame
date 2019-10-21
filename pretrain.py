import numpy as np
from Environment import *

def init_with_place(bomber,monster):
    matrix=np.asarray([[1,1,1,1,1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,1,0,0,0,1],[1,0,0,0,0,0,0,1,0,0,0,1],
                     [1,0,0,0,1,1,1,0,0,0,0,1],[1,1,0,0,0,0,1,0,0,1,1,1],[1,0,0,0,0,0,0,0,0,0,0,1],
                     [1,0,0,0,1,1,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1,0,1],[1,0,0,0,0,0,0,0,0,1,0,1],
                     [1,1,1,1,1,1,1,0,1,0,0,1],[1,0,0,0,0,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1,1,1,1,1]]).astype('float32')
    matrix[bomber[0],bomber[1]]=BOMBER_VALUE
    for i in monster:
        matrix[i[0],i[1]]=MONSTER_VALUE

    return matrix.reshape(1,12,12)

def expert():
    bomber_list=[(8,1),(8,2),(7,1),(7,2),(7,3),(6,2)]+[(8,7),(7,6),(7,7),(7,8),(6,6),(6,8)]+[(1,2),(2,1),(2,2),(2,3),(3,2),(3,3)]+[(1,9),(2,8),(2,9),(2,10),(3,9),(3,10)]
    bomber_list.extend([(6,5),(6,6),(7,6),(5,7),(7,7),(6,8)])
    bomber_list.extend([(6,1),(7,1),(7,2),(8,2),(7,3),(6,3)])
    bomber_list.extend([(8,1),(7,1),(7,2),(8,3),(7,3),(8,2)])
    monster_list=[[(8,3),(6,3),(6,1)]]*6+[[(8,6),(8,8),(6,7)]]*6+[[(1,1),(1,3),(3,1)]]*6+[[(1,8),(1,10),(3,8)]]*6
    monster_list.extend([[(5,5),(7,5),(6,7)]]*6)
    monster_list.extend([[(8,1),(8,3),(6,2)]]*6)
    monster_list.extend([[(6,1),(6,3),(8,2)]]*6)
    action_return=[4]
    action_return.extend([4]*len(bomber_list))

    trajectory=np.asarray([[1,1,1,1,1,1,1,1,1,1,1,1],[1,0,0,0.67,0.33,0,0.67,1,0,0,0,1],[1,0,0,0,0,0,0.67,1,0,0,0,1],
                     [1,0,0,0,1,1,1,0,0,0,0,1],[1,1,0,0,0,0,1,0,0,1,1,1],[1,0,0,0,0,0,0,0,0,0,0,1],
                     [1,0,0,0,1,1,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,1,0,1],[1,0,0,0,0,0,0,0,0,1,0,1],
                     [1,1,1,1,1,1,1,0,1,0,0,1],[1,0,0,0,0,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1,1,1,1,1]]).astype('float32')
    trajectory=trajectory.reshape((1,12,12))
    for i in range(len(bomber_list)):
        trajectory=np.concatenate((trajectory,init_with_place(bomber_list[i],monster_list[i])),axis=0)
    print(trajectory.shape)
    return trajectory,np.asarray(action_return)
