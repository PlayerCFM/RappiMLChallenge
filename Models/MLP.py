from sklearn.neural_network import MLPClassifier
import pandas as pd
from datetime import datetime

from Models.ModelInterface import ModelInterface

class MultiLayerPerceptronModel(ModelInterface):

    model = None

    def __init__(self, modelParams={}):
        self.model = MLPClassifier(**modelParams)

    def Fit(self, trainData, trainTarget):
        self.model.fit(trainData, trainTarget)

    def Predict(self, test_x):
        return self.model.predict(test_x)

    def GridSearch(self):
        return