from enum import Enum
from Models import GB, MLP

class SupportedModels(Enum):
    GradientBoostingModel = GB.GradientBoostingModel
    MultiLayerPerceptronModel = MLP.MultiLayerPerceptronModel