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
        reward = -10000
    
    done = stepR.state != 0

    return (np.reshape(state,[1, 30]), reward, done)

class Agent:
    def __init__(self):
        self.state_size = 30
        self.action_size = 4
        self.epsilon = 0.15
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.memory = deque(maxlen = 2000)
        self.tau = 0.125
        self.model = self.build_model()
        self.target_model = self.build_model()

    def chooseAction(self, state):
        if np.random.ranf() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = 'mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def replay(self, batchSize):
        minibatch = random.sample(self.memory, batchSize)

        for state,action,reward,next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                #järgmise käigu actionite ennustatavad rewardid
                a = self.model.predict(next_state)[0]
                #järgmise käig uactionite ennustatavad rewardid target_modeli järgi, loss jaoks
                t = self.target_model.predict(next_state)[0]
                #selle sammu parim käik oli käik, kus praegune reward ja tulevased rewardid oli kõige suuremad
                #argmax(a) on järgmise sammu parim käik
                #t[np.argmax(a)] on target modeli käik
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def loadModel(self, name):
        self.model.load_weights(name)
    
    def saveModel(self, name):
        self.model.save_weights(name)

EPISODES = 200001
agent = Agent()
#agent.loadModel('model_after_145000.h5')


reward_t_history = []
reward_history = []
winCount = 0
winrate_history = [] 
fullGameCount = 0
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

    for step_number in range(1000):
        action = agent.chooseAction(state)

        game = game.step(action)
        next_state, reward, done =  formStepResult(game)

        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            fullGameCount+=1
            if(game.player1Score > game.player2Score):
                winCount+=1 
            print("ep: {}, score: {}-{}, exploration rate: {}". format(e, game.player1Score, game.player2Score, agent.epsilon))
            break
    

    
    agent.replay(32)
    agent.update_target();
    #print("ep: {}, score: {}-{}, total reward : {}". format(e, game.player1Score, game.player2Score, t_reward))
    #reward_t_history.append(game.player1Score - game.player2Score);

    '''
    if(e % 100 == 0 and e != 0):
        num = sum(reward_t_history)/len(reward_t_history)
        reward_history.append(num)
        reward_t_history=[]
        print(num)
        '''
    

    if(e % 1000 == 0 and e != 0):
       ''''
       agent.saveModel('model_after_' + str(e + 64000) + '.h5')
       plt.plot(reward_history)
       plt.savefig('graph_' + str(e))
       '''
       winRate = winCount/fullGameCount
       print(winRate)
       winrate_history.append(winRate)
       winCount = 0
       fullGameCount = 0
    
    if(e % 10000 == 0 and e != 0):
        agent.saveModel('model_after_' + str(e) + '.h5')
        plt.plot(winrate_history)
        plt.savefig('graph_wr_' + str(e))
        print(winrate_history)


plt.plot(winrate_history)
print(winrate_history)
plt.savefig('graph_' + str(e))