import numpy as np
from  Environment import *
from Agent import *
from pretrain import  *

EPISODES=30*10000
PRE_TRAIN=False
PRE_EPOCH=4

def sample(distribution):
    return np.random.choice(distribution.shape[0],p=distribution)


def game(environment,agent):
    action_list = ['up', 'down', 'left', 'right', 'boom']
    count =0
    while True:
        count += 1

        # bomber走两步
        state = environment.get_state()
        action = sample(agent.action(state))
        result = environment.action(action_list[action])

        # print(environment.bomber)
        # print(environment.monster)

        if result == 'fail':
            trajectory,actions,num,is_bomb=environment.get_record()
            print("you finally failed")
            return trajectory, actions, reward_func(num, len(trajectory), is_bomb),environment.get_demo_traj()
        elif result == 'win':
            trajectory, actions,num ,is_bomb= environment.get_record()
            print("you finally won the game!")
            return trajectory, actions, reward_func(num, len(trajectory), is_bomb),environment.get_demo_traj()

        state = environment.get_state()
        action = sample(agent.action(state))
        result = environment.action(action_list[action])

        # print(environment.bomber)
        # print(environment.monster)

        if result == 'fail':
            trajectory, actions, num, is_bomb = environment.get_record()
            print("you finally failed")
            return trajectory, actions, reward_func(num, len(trajectory), is_bomb),environment.get_demo_traj()
        elif result == 'win':
            trajectory, actions, num, is_bomb = environment.get_record()
            print("you finally won the game!")
            return trajectory, actions, reward_func(num, len(trajectory), is_bomb),environment.get_demo_traj()

        # monster走一步
        result = environment.update_state()
        # print(environment.bomber)
        # print(environment.monster)

        if result == 'fail':
            trajectory, actions, num, is_bomb = environment.get_record()
            print("you finally failed")
            return trajectory, actions, reward_func(num, len(trajectory), is_bomb),environment.get_demo_traj()
        elif result == 'win':
            trajectory, actions, num, is_bomb = environment.get_record()
            print("you finally won the game!")
            return trajectory, actions, reward_func(num, len(trajectory), is_bomb),environment.get_demo_traj()

        if count>500:
            trajectory, actions ,num,is_bomb= environment.get_record()
            print("too long time, you lose")
            return trajectory, actions, -1,environment.get_demo_traj()


if __name__ == '__main__':
    agent = Agent()
    list_reward=[]
    if PRE_TRAIN:
        pre_set,pre_label=expert()
        agent.bhv_clone(pre_set,pre_label,PRE_EPOCH)

    action_list = ['up', 'down', 'left', 'right', 'boom']
    vectorization = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'boom': 4}

    for i in range(EPISODES):
        environment=Environment()
        print("episodes "+str(i)+" start:")
        trajectory,actions,reward,_=game(environment,agent)
        print("reward on this episode: "+str(reward))
        list_reward.append(reward)
        actions=list(map(lambda x:vectorization.get(x),actions))
        print(environment.bomber)
        print(environment.monster)
        agent.train(trajectory,actions,reward)
    agent.save()
    np.savetxt("log.txt",np.asarray(list_reward))
    # matrix = np.asarray([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    #                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    #                    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    #                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    #                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    #                    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # environment = Environment(matrix.astype('float32'))
    # state = environment.get_state()
    # print(state)
    # print(environment.bomber)
    # print(environment.monster)
    # action = "up"
    # result = environment.action(action)
    # print(result)
    # print(environment.get_state())
    # print(environment.bomber)
    # print(environment.monster)
    # action='left'
    # result = environment.action(action)
    # print(result)
    # print(environment.get_state())
    # print(environment.bomber)
    # print(environment.monster)
    # result=environment.update_state()
    # print(result)
    # print(environment.get_state())
    # print(environment.bomber)
    # print(environment.monster)
    # print(environment.train_trajectory[0],environment.train_trajectory[1])


