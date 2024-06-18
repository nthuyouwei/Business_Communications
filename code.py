# Monte Carlo
# this code is used to generate a trajectory using monte carlo or probability emthod
# with 2 trajectory storage capacity
# we first generate a random movement and check if the movement is legal by checking
# if it hits the wall or obstacle
# if it pass the test, we execute it by changing state variable

import random

width=10
length=10

def insidewindow(x,y):
    if x<width and x>=0:
        if y<length and y>=0:
            return True
    else:
        return False
def obstacle(x,y):
    if x==5 and y==5:
        return False # hitting the obstacle
    elif x==5 and y==6:
        return False
    elif x==5 and y==7:
        return False
    elif x==5 and y==8:
        return False
    elif x==5 and y==9:
        return False
    elif x==5 and y==0:
        return False
    elif x==5 and y==1:
        return False
    elif x==5 and y==2:
        return False
    elif x==5 and y==3:
        return False
    else:
        return True

prev_trajectory= [0,0,0,0,0,0,0,0,0,0,0, \
                0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0]

current_trajectory= [0,0,0,0,0,0,0,0,0,0,0, \
                0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0,\
                     0,0,0,0,0,0,0,0,0,0,0]

PI= [0.6,0.2,0.2] #probability of moving up, moving right, and moving left

count_move=0  # counting of legal move
count_move_prev=100 # record of best move minimal no of move
for k in range(1000):
    state_x=0
    state_y=0
    count_move=0
    for l in range(100):
            current_trajectory[l]=0 #make all item to zero
    for i in range(100):
        p=random.uniform(0,1) #generate another random number
        if p>=0 and p<=PI[0]:
            state_y=state_y+1  # MOVE UP
            if insidewindow(state_x,state_y) and obstacle(state_x,state_y):
              #  print('move up')
              #  print('x,y', state_x, state_y)
                current_trajectory[count_move]= 1
                count_move=count_move+1
            else:
                state_y=state_y-1

        elif p>PI[0] and p<=(PI[0]+PI[1]):
            state_x=state_x+1 #move right
            if insidewindow(state_x,state_y)and obstacle(state_x,state_y):
               # print('move right')
               # print('x,y', state_x, state_y)
                current_trajectory[count_move]= 2
                count_move=count_move+1
            else:
                state_x=state_x-1
        else:
            state_x=state_x-1 #move left
            if insidewindow(state_x,state_y)and obstacle(state_x,state_y):
               # print('move left')
              #  print('x,y', state_x, state_y)
                current_trajectory[count_move]= 3
                count_move=count_move+1
            else:
                state_x=state_x+1
        if state_x==width-1 and state_y==length-1:
            print('iteration',k, 'move', count_move, 'goal reached')
            break
    if count_move<count_move_prev: #if the current move is better
        count_move_prev=count_move
        for j in range(100):
            prev_trajectory[j]=current_trajectory[j] #store the move history
    if k % 100==0:  #sample result of improvement
        print('iter', k, 'result', count_move_prev)

for i in range(100):
    if prev_trajectory[i] !=0:

        print('index', i, 'movement', prev_trajectory[i])



############ MDP  1 #############


import numpy as np

def run_value_iteration(states, actions, rewards, transition, discount_factor=0.9, theta=0.0001):
    V = np.zeros(len(states))  # Initialize all state values to zero
    policy = np.zeros((len(states), len(actions)))  # Initialize policy

    while True:
        delta = 0
        for s in states:
            v = V[s]
            # Update each state's value based on the best possible action
            V[s] = max(sum([transition[s][a][s_prime] * (rewards[s][a] + discount_factor * V[s_prime])
                            for s_prime in states]) for a in actions)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    for s in states:
        # Update policy to perform the best action in each state
        action_values = np.array([sum([transition[s][a][s_prime] * (rewards[s][a] + discount_factor * V[s_prime])
                                       for s_prime in states]) for a in actions])
        best_action = np.argmax(action_values)
        policy[s] = np.eye(len(actions))[best_action]

    return policy, V

# Example usage for a very simple scenario:
states = list(range(100))  # States are indexed from 0 to 99
actions = ['up', 'right', 'down', 'left']  # Possible actions
rewards = np.full((100, 4), -1.0)  # Default reward is -1
transition = {}  # Define your transition probabilities here

# Add custom logic for transitions and rewards based on your grid's specifics

# Run Value Iteration
policy, values = run_value_iteration(states, actions, rewards, transition)

print("Optimal Policy:", policy)
print("State Values:", values)


############ MDP  2 #############


import numpy as np

width = 10
length = 10

actions = ['up', 'down', 'left', 'right']
action_vectors = {
    'up': (0, 1),
    'down': (0, -1),
    'left': (-1, 0),
    'right': (1, 0)
}

# 定義獎勵矩陣和狀態轉移矩陣
rewards = np.full((width, length), -1)  # 每次移動的獎勵
rewards[9, 9] = 100  # 目標位置的獎勵

obstacles = [(5, i) for i in range(10)]
for (x, y) in obstacles:
    rewards[x, y] = -100  # 障礙物位置的獎勵

def is_inside(x, y):
    return 0 <= x < width and 0 <= y < length

def get_next_state(x, y, action):
    dx, dy = action_vectors[action]
    next_x, next_y = x + dx, y + dy
    if is_inside(next_x, next_y) and rewards[next_x, next_y] != -100:
        return next_x, next_y
    return x, y

# 值迭代
values = np.zeros((width, length))
gamma = 0.9  # 折扣因子

for _ in range(1000):
    new_values = np.copy(values)
    for x in range(width):
        for y in range(length):
            action_values = []
            for action in actions:
                next_x, next_y = get_next_state(x, y, action)
                action_values.append(rewards[next_x, next_y] + gamma * values[next_x, next_y])
            new_values[x, y] = max(action_values)
    values = new_values

# 提取策略
policy = np.empty((width, length), dtype=str)
for x in range(width):
    for y in range(length):
        action_values = []
        for action in actions:
            next_x, next_y = get_next_state(x, y, action)
            action_values.append((rewards[next_x, next_y] + gamma * values[next_x, next_y], action))
        policy[x, y] = max(action_values)[1]

# 顯示結果
print("Optimal Policy:")
for y in range(length):
    for x in range(width):
        print(policy[x, y], end=' ')
    print()


########## Q-learning  ######

import numpy as np
import random

width, length = 10, 10
actions = ['up', 'down', 'left', 'right']
action_vectors = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}

# 環境設定
obstacles = [(5, i) for i in range(10)]
rewards = np.full((width, length), -1.0)
for obs in obstacles:
    rewards[obs] = -100
rewards[(9, 9)] = 100

# Q 表初始化
Q = np.zeros((width, length, len(actions)))

# 參數設定
epsilon = 0.1  # 探索率
alpha = 0.5  # 學習率
gamma = 0.9  # 折扣因子
episodes = 5000  # 迭代次數

def choose_action(x, y):
    if random.random() < epsilon:
        return random.choice(actions)  # 探索
    else:
        return actions[np.argmax(Q[x, y])]  # 利用

def update_environment(x, y, action):
    if (x, y) == (9, 9):
        return x, y, 0  # 終點獎勵
    next_x, next_y = x + action_vectors[action][0], y + action_vectors[action][1]
    if not (0 <= next_x < width and 0 <= next_y < length) or (next_x, next_y) in obstacles:
        return x, y, -100  # 遇到障礙或邊界
    return next_x, next_y, rewards[next_x, next_y]

# Q-learning 迭代過程
for episode in range(episodes):
    x, y = 0, 0  # 起點
    while (x, y) != (9, 9):
        action = choose_action(x, y)
        next_x, next_y, reward = update_environment(x, y, action)
        best_next_action = np.argmax(Q[next_x, next_y])
        Q[x, y, actions.index(action)] += alpha * (reward + gamma * Q[next_x, next_y, best_next_action] - Q[x, y, actions.index(action)])
        x, y = next_x, next_y

# 打印結果
for y in range(length):
    for x in range(width):
        print(actions[np.argmax(Q[x, y])], end=' ')
    print()



############## Sarsa ########

import numpy as np
import random

width, length = 10, 10
actions = ['up', 'down', 'left', 'right']
action_vectors = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}

# 環境設定
obstacles = [(5, i) for i in range(10)]
rewards = np.full((width, length), -1.0)
for obs in obstacles:
    rewards[obs] = -100
rewards[(9, 9)] = 100

# Q 表初始化
Q = np.zeros((width, length, len(actions)))

# 參數設定
epsilon = 0.1  # 探索率
alpha = 0.5  # 學習率
gamma = 0.9  # 折扣因子
episodes = 5000  # 迭代次數

def choose_action(x, y, q_values):
    if random.random() < epsilon:
        return random.choice(actions)  # 探索
    else:
        return actions[np.argmax(q_values)]  # 利用

# SARSA 迭代過程
for episode in range(episodes):
    x, y = 0, 0  # 起點
    current_action = choose_action(x, y, Q[x, y])
    while (x, y) != (9, 9):
        next_x, next_y = x + action_vectors[current_action][0], y + action_vectors[current_action][1]
        if not (0 <= next_x < width and 0 <= next_y < length) or (next_x, next_y) in obstacles:
            next_x, next_y = x, y  # 若移動無效，則保持原位
            reward = -100
        else:
            reward = rewards[next_x, next_y]
        
        next_action = choose_action(next_x, next_y, Q[next_x, next_y])
        Q[x, y, actions.index(current_action)] += alpha * (reward + gamma * Q[next_x, next_y, actions.index(next_action)] - Q[x, y, actions.index(current_action)])
        
        x, y = next_x, next_y
        current_action = next_action

# 打印結果
for y in range(length):
    for x in range(width):
        print(actions[np.argmax(Q[x, y])], end=' ')
    print()
