import pandas as pd
from joblib import dump, load
import importlib
from datetime import datetime

import Models
from Utils.Settings import ProjectSettings

class ManageModels:
    def GetSavedModels(self):
        return pd.read_csv(ProjectSettings.Settings()['SavedModelsLocationDB'])
    
    def LoadModelAndData(self, modelInfo):
        model_module = importlib.import_module(modelInfo["ModelModule"])
        model_class = getattr(model_module,modelInfo["ModelType"])()
        model_class.model = load(modelInfo["ModelSavedAt"])

        data_module = importlib.import_module(modelInfo["DataModule"])
        data_class = getattr(data_module,modelInfo["DataType"])()
        data_class.scaler = load(modelInfo["ScalerSavedAt"])

        return model_class, data_class

    def SaveModel(self, model, loader, metrics={}):
        scaler = loader.scaler
        models_csv = pd.read_csv(ProjectSettings.Settings()['SavedModelsLocationDB'])
        model_params = model.model.get_params()

        saved_date = datetime.now()
        model_saved_path = ProjectSettings.Settings()['SavedModelsLocation']
        scaler_saved_path = ProjectSettings.Settings()['SavedScalersLocation']
        model_saved_name = f'{type(model).__name__}-{saved_date}.joblib'
        scaler_saved_name = f'{type(scaler).__name__}-{saved_date}.joblib'

        training_information = {
            "ModelType": type(model).__name__,
            "ModelModule": model.__module__,
            "DataType": type(loader).__name__,
            "DataModule": loader.__module__,
            "HourSaved": saved_date,
            "F1Score": metrics["F1"] if "F1" in metrics else None,
            "ModelSavedAt": model_saved_path+model_saved_name,
            "ScalerSavedAt": scaler_saved_path+scaler_saved_name
        }

        model = model.model

        dump(model, model_saved_path+model_saved_name) 
        dump(scaler, scaler_saved_path+scaler_saved_name) 

        pd.concat([
            models_csv,
            pd.DataFrame([training_information])
            ]).to_csv(ProjectSettings.Settings()['SavedModelsLocationDB'], index=False)