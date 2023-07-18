import os
import sys

from DataLoader.Loader import TitanicData
from Core.Pipeline import Pipeline
from Models.ManageModels import ManageModels
from Models import SupportedModels
from DataLoader.SupportedData import SupportedData
from Core.Metrics import Metrics

from Utils.Settings import ProjectSettings

import pandas as pd

class CLI:

    def __init__(self):
        self.cli = True

    def ClearTerminal(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def ShowMainMenu(self):

        choice = -1

        while True:
            self.ClearTerminal()
            print("Show an option to continue:")
            print("\t1) Show Training and Testing Data")
            print("\t2) List saved models")
            print("\t3) Run training pipeline")
            print("\t4) Load Model and Data")
            print("\t10) Exit")

            choice = int(input())

            if choice == 1:
                self.ShowData()
                print("Press Any key to continue")
        
            elif choice == 2:
                self.ListSavedModels()
                print("Press Any key to continue")
            
            elif choice == 3:
                self.RunTrainingPipeline()
                print("Press Any key to continue")

            elif choice == 4:
                self.LoadModel()
                print("Press Any key to continue")
        
            elif choice == 10:
                self.ClearTerminal()
                print("Good bye!")
                sys.exit(0)
            
            else:
                self.ClearTerminal()
                print("You did not choose a correct option")

            input()
            choice = -1

    def ShowData(self):
        self.ClearTerminal()
        titanicData = TitanicData()
        trainData = titanicData.Get_TrainData()
        testData = titanicData.Get_TestData()
        print("***********************Training Data*****************************")
        print(trainData)

        print("\n\n\n\n\n***********************Testing Data*****************************")
        print(testData)
    
    def ListSavedModels(self):
        self.ClearTerminal()
        modelsManager = ManageModels()
        savedModels = modelsManager.GetSavedModels()
        
        print("***********************Saved Models*****************************")
        print(savedModels)

    def RunTrainingPipeline(self):
        self.ClearTerminal()
        print("""This option will start a new training from scratch...
              \n\n\nWhat type of model do you like to train?
              """)
        
        supported_models_info = []

        for i, model_type in enumerate(SupportedModels.SupportedModels):
            print(f"\t{i}) {model_type.name}")
            supported_models_info.append({
                "model_type": model_type.name
            })

        supported_models_info = pd.DataFrame(supported_models_info)

        model_id = int(input())

        model_type = SupportedModels.SupportedModels[
            supported_models_info.iloc[model_id]['model_type']
        ].value
        
        print("""\n\n\nWhat dataset do you like to use?""")
        supported_data_info = []

        for i, data_type in enumerate(SupportedData):
            print(f"\t{i}) {data_type.name}")
            supported_data_info.append({
                "data_type": data_type.name
            })

        supported_data_info = pd.DataFrame(supported_data_info)

        data_id = int(input())

        data_type = SupportedData[
            supported_data_info.iloc[data_id]['data_type']
        ].value

        ###################### Run the pipeline
        self.ClearTerminal()

        data_loader, trained_model = Pipeline().RunPipeline(data_type, model_type)

        print("\n\n\nWould you like to see model's performance? [y/n]")
        performance_model_answer = str(input())
        if performance_model_answer=="y":
            self.ClearTerminal()
            print("***********************Metrics Results*****************************")
            pred_y = trained_model.Predict(data_loader.test_x)

            metrics_result = Metrics().ClassificationMetrics(data_loader.test_Y, pred_y)
            for i, metric in enumerate(metrics_result):
                print(f"\n\n{metric}: \n{metrics_result[metric]}")

        print("\n\n\nWould you like to save the model? [y/n]")
        save_model_answer = str(input())
        if save_model_answer=="y":
            print("Saving model...")
            modelsManager = ManageModels()
            savedModels = modelsManager.SaveModel(trained_model, data_loader)

    def LoadModel(self):
        self.ClearTerminal()
        modelsManager = ManageModels()
        savedModels = modelsManager.GetSavedModels()

        print(savedModels)

        print("\n\nChoose the ID of the model you like to load")
        modelId = int(input())
        selectedModel = savedModels.iloc[modelId]

        model, data_loader = modelsManager.LoadModelAndData(selectedModel)

        print(f"Model ({selectedModel.ModelType}) and data were loaded ({selectedModel.DataType})")
        train_x, train_Y, test_x, test_Y = data_loader.RunPreprocessPipeline()

        pred_y = model.Predict(test_x)

        print(f"\n\n\nWould you like to see model's performance? (it will use the preprocessed {selectedModel.DataType} data) [y/n]")
        performance_model_answer = str(input())
        if performance_model_answer=="y":
            self.ClearTerminal()
            print("***********************Metrics Results*****************************")
            pred_y = model.Predict(data_loader.test_x)

            metrics_result = Metrics().ClassificationMetrics(data_loader.test_Y, pred_y)
            for i, metric in enumerate(metrics_result):
                print(f"\n\n{metric}: \n{metrics_result[metric]}")

        print(f"\n\n\nWould you like Predict on a random data point? (from the {selectedModel.DataType} data) [y/n]")
        performance_model_answer = str(input())
        if performance_model_answer=="y":
            self.ClearTerminal()
            print("***********************Random Sample*****************************")
            random_sample_x = data_loader.test_x.sample()
            random_sample_Y = data_loader.test_Y.iloc[random_sample_x.index]

            print(f"\n\nThis is the randomly picked sample: \n {random_sample_x}")
            print(f"\nPassenger belongs to the {int(random_sample_Y)} class")

            pred_y = model.Predict(random_sample_x)

            print(f"\nThe model predicted {pred_y[0]} class")

            print(f"\nmodel prediction was {'CORRECT' if int(random_sample_Y)==pred_y[0] else 'INCORRECT'}")

cli = CLI()
cli.ShowMainMenu()