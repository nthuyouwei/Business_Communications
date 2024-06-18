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
