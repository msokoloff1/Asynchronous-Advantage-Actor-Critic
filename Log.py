from multiprocessing import Value
 
class Logger():
    def __init__(self,tf, logPath, level = 'VERBOSE', logFrequency = 1000):
        #All logging goes through this class
        #Allows for global control in one location
        self.logPath = logPath
        self.tf = tf
        self.level = level
        self.logFrequency = logFrequency
        self.avgRewards = Value('d',0.0, lock = False)
        self.gamesPlayed = Value('i', 0, lock = False)
        
    
    def addTurnReward(self, reward, id):
        if(int(id) == 0):
            print("reward - %s"%str(reward))
        self.avgRewards.value = (0.99*self.avgRewards.value + 0.01*reward)
        self.gamesPlayed.value += 1
        
        
           
    def getSummaryOp(self):
        return self.summary_op
            
    def addSummary(self, name, tensor):
        self.tf.summary.scalar(name, tensor)
        
    def addHist(self, name, tensor):
        self.tf.summary.histogram(name, tensor)
        
    def writeSummaries(self, summaryEvaluated, iter):
        self.writer.add_summary(summaryEvaluated, iter)
        
    def addNonTensorSummary(self, name, value):
        self.tf.Summary(value=[self.tf.Summary.Value(tag=name, simple_value=value)])
        
    def ready(self, sess):
        self.summary_op = self.tf.summary.merge_all()
        self.writer = self.tf.summary.FileWriter(self.logPath)
        