from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from datetime import datetime
from joblib import dump, load

from Models.ModelInterface import ModelInterface

class GradientBoostingModel(ModelInterface):

    model = None

    def __init__(self, modelParams={}):
        self.model = GradientBoostingClassifier(**modelParams)

    def Fit(self, trainData, trainTarget):
        self.model.fit(trainData, trainTarget)

    def Predict(self, test_x):
        return self.model.predict(test_x)
    
    def GridSearch(self):
        return