from abc import ABC, abstractmethod

class ModelInterface(ABC):

    @property
    @abstractmethod
    def model(self):
        pass
    
    @abstractmethod
    def Fit(self, trainData, trainTarget):
        pass

    @abstractmethod
    def Predict(self, test_x):
        pass
    
    def GetMetrics(self):
        return
        
    @abstractmethod
    def GridSearch(self):
        pass