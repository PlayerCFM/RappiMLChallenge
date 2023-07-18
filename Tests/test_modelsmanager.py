from Models.ManageModels import ManageModels
import unittest
from unittest.mock import patch

class TestModelsManager(unittest.TestCase):
    
    def test_LoadModelAndData(self):
        with self.assertRaises(ModuleNotFoundError):
            ManageModels().LoadModelAndData(
                {
                    "ModelModule": "example_module", 
                    "ModelType": "example_class",
                    "ModelSavedAt": "/fake_directory"
                }
            )

        with self.assertRaises(AttributeError):
            ManageModels().LoadModelAndData(
                {
                    "ModelModule": "Models.GB", 
                    "ModelType": "example_class",
                    "ModelSavedAt": "/fake_directory"
                }
            )

        with self.assertRaises(FileNotFoundError):
            ManageModels().LoadModelAndData(
                {
                    "ModelModule": "Models.GB", 
                    "ModelType": "GradientBoostingModel",
                    "ModelSavedAt": "/fake_directory"
                }
            )

    def test_SaveModel(self):
        with self.assertRaises(ModuleNotFoundError):
            ManageModels().LoadModelAndData(
                {
                    "ModelModule": "example_module", 
                    "ModelType": "example_class",
                    "ModelSavedAt": "/fake_directory"
                }
            )

        with self.assertRaises(AttributeError):
            ManageModels().LoadModelAndData(
                {
                    "ModelModule": "Models.GB", 
                    "ModelType": "example_class",
                    "ModelSavedAt": "/fake_directory"
                }
            )

        with self.assertRaises(FileNotFoundError):
            ManageModels().LoadModelAndData(
                {
                    "ModelModule": "Models.GB", 
                    "ModelType": "GradientBoostingModel",
                    "ModelSavedAt": "/fake_directory"
                }
            )
        



        