from abc import ABC, abstractmethod
import random
import pandas

from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load
from datetime import datetime

from DataLoader.SupportedScalers import SupportedScalers

class SupportedScalers(Enum):
    Standard = StandardScaler
    MinMax = MinMaxScaler

class Preprocess(ABC):

    def __init__(self, scaler: SupportedScalers=SupportedScalers.Standard):
        self.scaler = scaler

    @abstractmethod
    def RunPreprocessPipeline(self):
        pass

    @abstractmethod
    def PreprocessTrainData(self):
        pass

    @abstractmethod
    def PreprocessTestData(self):
        pass

    def FitScaler(self, data, featuresToScale: list = None, selectedScaler: SupportedScalers = SupportedScalers.Standard):
        
        self.scaler = selectedScaler.value()

        if not featuresToScale:
            featuresToScale=data.columns
        self.scaler.fit(data[featuresToScale])

    def ScaleData(self, data, featuresToScale: list = None):
        if not featuresToScale:
            featuresToScale=data.columns

        data[featuresToScale] = self.scaler.transform(data[featuresToScale])
        return data