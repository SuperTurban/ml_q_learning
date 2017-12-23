from game import Game
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import random
import time

import matplotlib.pyplot as plt

#mitu mängu mängida
#state [p1x, p1y, p1xs, p1ys, p1r, 
#       p2x, p2y, p2xs, p2ys, p2r, 
#       b1x, b1y, b1xs, b1ys,
#       b2x, b2y, b2xs, b2ys,
#       b3x, b3y, b3xs, b3ys,
#       b4x, b4y, b4xs, b4ys,
#       b5x, b5y, b5xs, b5s ]

#       Q function
#       käigu reward + järgmise 30 käigu reward 
#       
#       vaja salvestada võtta memorist 30 state, minibactch koosneb id-dest

def formStepResult(stepR):
    state = [stepR.player1.x, stepR.player1.y, stepR.player1.speed_x, stepR.player1.speed_y, stepR.player1.rotation,
    stepR.player2.x, stepR.player2.y, stepR.player2.speed_x, stepR.player2.speed_y, stepR.player2.rotation]

    for i in range(5):
        if len(stepR.balls) > i:
           state.extend([stepR.balls[i].x, stepR.balls[i].y, stepR.balls[i].speed_x, stepR.balls[i].speed_y]) 
        else:
            state.extend([0,0,0,0])


    reward = 0 
    if stepR.player1scored == 1:
        reward = 10000
    elif stepR.player2scored == 1:
        reward = -11000
    
    done = stepR.state != 0

    return (np.reshape(state,[1, 30]), reward, done)

class Agent:
    def __init__(self):
        self.state_size = 30
        self.action_size = 4
        self.epsilon = 0.25
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.memory = deque(maxlen = 2000)
        self.model = self.build_model()


    def chooseAction(self, state):
        if np.random.ranf() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        print(act_values)
        return np.argmax(act_values[0])
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = 'mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def replay(self, batchSize):
        minibatch = random.sample(range(len(self.memory)), batchSize)

        # 30 järgmist käiku
        # 30 liida rewardid kokku
        # võta käik, mille reward on kõike suurem
        # treeni mudelit seda käiku valima
        for index in minibatch:
            state,action,reward,next_state, done = self.memory[i]
            target = reward

            if not done:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs = 1, verbose = 0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def loadModel(self, name):
        self.model.load_weights(name)
    
    def saveModel(self, name):
        self.model.save_weights(name)

EPISODES = 10
agent = Agent()

reward_t_history = []
reward_history = []

for e in range(EPISODES):
    useUI = False
    player1AI = True

    '''
    if e == 100:
        useUI = True
        player1AI = False
    '''

    game = Game(useUI, player1AI)
    state, reward, done = formStepResult(game)
    t_reward = 0

    for step_number in range(2000):
        action = agent.chooseAction(state)

        game = game.step(action)
        next_state, reward, done =  formStepResult(game)

        agent.memory.append((state, action, reward, next_state, done))
        t_reward += reward
        state = next_state

        if done:
            print("ep: {}, score: {}-{}, total reward : {}". format(e, game.player1Score, game.player2Score, t_reward))
            break
    
    agent.replay(32)
    print("ep: {}, score: {}-{}, total reward : {}". format(e, game.player1Score, game.player2Score, t_reward))
    reward_t_history.append(t_reward);

    if(e % 20 == 0):
        num = sum(reward_t_history)/len(reward_t_history)
        reward_history.append(num)
        reward_t_history=[]
        print(num)

    
    #if(e % 50 == 0 and e != 0):
    #    agent.saveModel('model_after_' + str(e) + '.h5')
    #

plt.plot(reward_history)
plt.show();
time.sleep(3)

