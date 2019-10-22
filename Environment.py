import numpy as  np
import random

BOMBER_VALUE=100
MONSTER_VALUE=-100

def reward_func(num,length,is_bomb):
    reward=(10**(3-num))/2
    if num ==3 : reward=-100
    reward=reward*((101/100)**(-length+1))

    if not is_bomb:
        reward=length/60.0-5
    return reward



def equal(x, y):
    eps = 1e-7
    return abs(x - y) <= eps

class Environment():

    def __init__(self):
        # 初始化，wall为1，空地为0
        self.matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                             [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                             [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        print("initializing environment")


        # 初始化bomber和monster位置
        self.bomber = self.init_position('bomber')
        self.monster=[self.init_position('monster') for i in range(3)]


        # 初始化state，画上bomber和monster,bomber为0.75，monster为0.25
        # self.matrix[self.bomber]=0.75
        # for i in self.monster:
        #     self.matrix[i]=0.25

        self.is_bomb=False

        # 初始化轨迹为空
        self.train_trajectory=[]
        self.actions=[]
        self.demo_trajectory=[]
        print("initializing finished!")

    def init_position(self,role):
        elige_list=[]
        for i in range(12):
            for j in range(12):
                if equal(self.matrix[i,j],0.0):
                    elige_list.append((i,j))
        random.shuffle(elige_list)
        i,j=elige_list[0]

        if role=="monster":
            self.matrix[i,j] = MONSTER_VALUE
        else :
            self.matrix[i,j] = BOMBER_VALUE
        return (i,j)

    def move(self,origin,role,direction='random'):
        # 根据角色类型移动，如果是monster就随机移动，bomber则需指定方向
        i, j = origin
        if role=='monster':
            value=MONSTER_VALUE
        else:
            value=BOMBER_VALUE

        # 建立上下左右的cursor，True表示有墙
        up_cursor=self.matrix[i-1,j]==1
        down_cursor=self.matrix[i+1,j]==1
        left_cursor=self.matrix[i,j-1]==1
        right_cursor=self.matrix[i,j+1]==1

        # direction 为 random 随机从没有墙的方向中选择一个
        if direction == 'random':
            select=np.where(1-np.asarray([up_cursor,down_cursor,left_cursor,right_cursor]))
            # print("this is select:")
            # print(select[0])
            index=np.random.choice(select[0])
            direction=['up','down','left','right'][index]
        # 移动目标：上，下，左，右，随机（monster）
        if direction=='up':
            if up_cursor:
                pass
            else:
                self.matrix[i-1,j]=value
                self.matrix[i,j]=0
                i=i-1
        elif direction=='down':
            if down_cursor:
                pass
            else:
                self.matrix[i+1,j]=value
                self.matrix[i, j] = 0
                i=i+1
        elif direction=='left':
            if left_cursor:
                pass
            else:
                self.matrix[i,j-1]=value
                self.matrix[i, j] = 0
                j=j-1

        elif direction == 'right':
            if right_cursor:
                pass
            else:
                self.matrix[i,j+1]=value
                self.matrix[i, j] = 0
                j=j+1

        else:
            assert False

        return i,j

    def bomb(self):
        self.is_bomb=True
        result='fail'
        i,j=self.bomber

        # 定义炸弹范围
        # affected_points = [(i,j+1),(i,j+2),(i,j-1),(i,j-2),(i+1,j),(i+2,j),(i-1,j),(i-2,j)]
        affected_points=[]
        i=i-2
        j=j-2
        for p in range(5):
            for q in range(5):
                affected_points.append((i+p,j+q))
        print("bomb:")
        print(affected_points)
        for p in  affected_points:
            if p in self.monster:
                self.monster.remove(p)
        if not self.monster:
            result = 'win'
        return result

    def action(self,action):
        result='wait'
        # 更新trajectory
        # print("train update with matrix:")
        # print(self.matrix)
        self.train_trajectory.append(self.matrix.copy())
        self.actions.append(action)

        # BOOM!
        if action=='boom':
            result=self.bomb()
            self.demo_trajectory.append(self.matrix)
            return  result

        # 移动bomber
        i,j=self.move(self.bomber,'bomber',action)
        self.demo_trajectory.append(self.matrix)
        self.bomber=(i,j)

        # print (equal(float(self.matrix[i, j]), MONSTER_VALUE))
        # 如果遇到monster，头铁撞上去，gameover
        if equal(float(self.matrix[i, j]), MONSTER_VALUE):
            result='failed'
        return result

    def get_state(self):
        return self.matrix

    def update_state(self):
        result = 'wait'
        # 轨迹中加入上一个state
        # print("demo update")
        self.demo_trajectory.append(self.matrix.copy())

        # 更新所有monster的位置
        for i in range(len(self.monster)):
            self.monster[i]=self.move(self.monster[i],'monster')

            # 如果monster和bomber位置重叠，gameover
            if self.monster[i]==self.bomber:
                result = 'fail'
        return result

    def get_record(self):
        return self.train_trajectory,self.actions,len(self.monster),self.is_bomb

    def get_demo_traj(self):
        return self.demo_trajectory

