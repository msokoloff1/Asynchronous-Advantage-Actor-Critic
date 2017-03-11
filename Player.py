from multiprocessing import Process, Queue
import Game
import numpy as np
import copy
import time
import random

class Player(Process):
    def __init__(self, gameType, log, imageDims, numFrames, trainingQueue, gamma,id ,memLen = 40):
        super().__init__()
        """
        gameType      : String containing the name of the OpenAI environment to be built
        log           : A reference to the global logging object
        imageDims     : The width and height for rescaling the observation
        numFrames     : Number of frames to stack in order to capture time dependencies
        trainingQueue : Shared memory to write experience to.
        gamma         : Discount factor to be applied to future rewards
        id            : The agents identifying number
        memLength     : Minimum length sequence to discount and learn from 
        """

        self.expAvgRewards = 0;
        self.log = log
        self.game = Game.Game(gameType, imageDims, numFrames)
        self.killed = False
        self.memLen = memLen
        self.gamma = gamma
        ##For chosing actions
        self.states = Queue(maxsize=1)
        self.result = Queue(maxsize=1)
        self.id = id
        self.epsilon = 0.0
        self.count = 0
        #For training the primary network (one global object that all learner classes use)
        self.trainingQueue = trainingQueue
        self.currentReward = 0.0
        self.epsilon = 1.0
        

    def run(self):
        """- Public method to initiate the trianing process """
        self.train()
        
    
    def kill(self):
        """ -Stops the training process """
        self.killed = True
    
    def train(self):
        """ - Manages the training process """ 
        while(not self.killed):
            self._play()
            self.game.reset()
          
    def _play(self):
        """ - Main game loop """

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
        """
        -Method that is called when the agent has reached a terminal state, used to update shared memory for learning

        stepsSinceUpdate : The number of steps since the last time the agent sent its memories to update the network
        memories         : Collection of observations required for learning
        """

        self.log.addTurnReward(self.currentReward, self.id)
        self.currentReward = 0.0
        memories['rewards'] = self._prepareRewards(memories['rewards'], 0.0)
        self.trainingQueue.put(memories)

        
    def _updater(self, stepsSinceUpdate, memories, value):
        """
        -Method that either increments the count since the last update or performs the update
         , used to update shared memory for learning

        stepsSinceUpdate : The number of steps since the last time the agent sent its memories to update the network
        memories         : Collection of observations required for learning
        value            : The value to be discounted backwards to the previous self.memLen steps
        """

        if(stepsSinceUpdate >= self.memLen):
            memories['rewards'] = self._prepareRewards(memories['rewards'], value)
            trainables  = {key:memories[key][:-1] for key in memories}
            memories    = {key:[memories[key][-1]] for key in memories}
            self.trainingQueue.put(trainables)
            
            stepsSinceUpdate = 0
        return [stepsSinceUpdate + 1, memories]

    def _prepareRewards(self, rewards, terminalValue):
        """
        - Makes the rewards at each step the observed reward plus the discounted future rewards 
             (future only lasting at most self.memLen time steps)

        rewards       : The observed rewards for each of the self.memLen number of timesteps
        terminalValue : The value of the last timestep observed. Either predicted or truly terminal  
        """

        reward_sum = terminalValue
        for index in reversed(range(0, len(rewards)-1)):
            r = np.clip(rewards[index], -1, 1)
            reward_sum = self.gamma * reward_sum + r
            rewards[index] = reward_sum
            
        return rewards
                        
