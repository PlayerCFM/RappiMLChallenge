from Core import Metrics, PreprocessInterface
from Models import GB

from Models.SupportedModels import SupportedModels

class Pipeline:
    
    def __init__(self):
        return
    
    def RunPipeline(self, dataLoader: PreprocessInterface.Preprocess, modelType: SupportedModels=SupportedModels.GradientBoostingModel):
        print(f"Starting Training Pipeline...")
        print(f"\n\tPreprocessing Data...")
        data_loader = dataLoader()
        train_x, train_Y, test_x, test_Y = data_loader.RunPreprocessPipeline()
        print(f"\n\tTraining the Model...")
        model = modelType()
        model.Fit(train_x, train_Y)
        y_pred = model.Predict(test_x)
        print(f"Training Pipeline Ended")

        return data_loader, model