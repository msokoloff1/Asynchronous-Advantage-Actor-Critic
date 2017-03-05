from Game import Game



class Demonstrator():
    def __init__(self,model, gameType, imageDims, numFrames):
        self.game = Game(gameType, imageDims, numFrames, monitor = True)
        self.model = model
        
    def play(self, numGames):
        
        while(numGames > 0):
            self.game.env.render()
            action, _ = self.model.predict(self.game._queueToNumpy(self.game.currentState))
            _,_,_,terminal,_ = self.game.step(np.argmax(action))
            
            if(terminal):
                numGames -= 1
                self.game.reset()
        