from multiprocessing import Process, Queue
import Game
import numpy as np
import copy
import time
import random

class Player(Process):
    def __init__(self, gameType, log, imageDims, numFrames, trainingQueue, gamma,id ,memLen = 5, numGames = 1):
        super().__init__()
        self.expAvgRewards = 0;
        self.log = log
        self.game = Game.Game(gameType, imageDims, numFrames)
        self.killed = False
        self.memLen = memLen
        self.gamma = gamma
        ##For chosing actions
        self.states = Queue(maxsize=numGames)
        self.result = Queue(maxsize=numGames)
        self.id = id
        self.epsilon = 0.0
        self.count = 0
        #For training the primary network (one global object that all learner classes use)
        self.trainingQueue = trainingQueue
        
        ###
        self.currentReward = 0.0
        self.epsilon = 1.0
        

    ###PUBLIC
    def run(self):
        self.train()
        
    
    def kill(self):
        self.killed = True
    
    def train(self):
        while(not self.killed):
            self._play()
            self.game.reset()
    ########################################################
          
    ###PRIVATE            
    def _play(self):
        
        stepsSinceUpdate = 0
        memories = {
              'rewards'     : []
            , 'prevStates'  : []
            , 'actions'     : []
        }

        while(True):
            self.states.put(self.game._queueToNumpy(self.game.currentState))
            actionProbs, value = self.result.get()
            action = np.random.choice(self.game.numActions, p=actionProbs)
            curState,prevState,  reward, terminal, info = self.game.step(action)
            
            memories['rewards'].append(reward)
            memories['prevStates'].append(prevState.reshape(1,84,84,4))
            memories['actions'].append(action)
        
            self.currentReward += reward
            
            if(terminal):                    
                self._handleTerminal(stepsSinceUpdate, memories)
                break
            
            stepsSinceUpdate, memories = self._updater(stepsSinceUpdate, memories, value)
            
    def _handleTerminal(self, stepsSinceUpdate, memories):
        self.log.addTurnReward(self.currentReward, self.id)
        self.currentReward = 0.0
        memories['rewards'] = self._prepareRewards(memories['rewards'], 0.0)
        self.trainingQueue.put(memories)

        
    def _updater(self, stepsSinceUpdate, memories, value):
        if(stepsSinceUpdate >= self.memLen):
            memories['rewards'] = self._prepareRewards(memories['rewards'], value)
            trainables  = {key:memories[key][:-1] for key in memories}
            memories    = {key:[memories[key][-1]] for key in memories}
            self.trainingQueue.put(trainables)
            
            stepsSinceUpdate = 0
        return [stepsSinceUpdate + 1, memories]

    def _prepareRewards(self, rewards, terminalValue):
        reward_sum = terminalValue
        for index in reversed(range(0, len(rewards)-1)):
            r = np.clip(rewards[index], -1, 1)
            reward_sum = self.gamma * reward_sum + r
            rewards[index] = reward_sum
            
        
        return rewards
                
            
            
            
        