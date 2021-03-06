from threading import Thread
import numpy as np

class Predictor(Thread):
    def __init__(self, agents, primaryNetwork):
        super().__init__()
        """
        agents         : A list of all agents that are training
        primaryNetwork : The network used for making predictions
        """

        self.setDaemon(True)
        self.primaryNetwork = primaryNetwork
        self.agents = agents
        self.killed = False
        
    def run(self):
        """
        - Controls when to stop the prediction process
        """
        while(not self.killed):
            self._predict()
            
    def _predict(self):
        """
        -Aggregates all observations from all agents to perform batch predictions
        """
            if(self.primaryNetwork.curIter > 100000000):
                for agent in self.agent:
                    agent.kill()
                    primaryNetwork.killAll = True
                    
                self._kill()
                
            queuedAgents = []
            states = []
            
            for agent in self.agents:
                try:
                    state = agent.states.get_nowait()
                    
                    queuedAgents.append(agent)
                    states.append(state.reshape(1,84,84,4))
                except:
                    pass
            
            if(len(queuedAgents) != 0):
                state = np.array(states)
                actionProbs, values = self.primaryNetwork.predict(state.reshape(-1,84,84,4))
                
                for agent, probs, vals in zip(queuedAgents, actionProbs, values):
                    agent.result.put([probs,vals])
                


    def _kill(self):
        """
        - Stops the prediction process
        """
        self.killed = True
