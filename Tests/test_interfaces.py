from Core.PreprocessInterface import Preprocess
from Models.ModelInterface import ModelInterface
import unittest

class TestPreProcessInterface(unittest.TestCase):

    def test_Abstraction(self):

        class ChildFromPreprocess(Preprocess):
            def __init__(self):
                pass

        with self.assertRaises(TypeError) as message:
            ChildFromPreprocess()
        self.assertEqual(str(message.exception), "Can't instantiate abstract class ChildFromPreprocess with abstract methods PreprocessTestData, PreprocessTrainData, RunPreprocessPipeline")

        class ChildFromPreprocess(Preprocess):
            def __init__(self):
                pass
            def PreprocessTestData(self):
                pass
            def PreprocessTrainData(self):
                pass
                
        with self.assertRaises(TypeError) as message:
            ChildFromPreprocess()
        self.assertEqual(str(message.exception), "Can't instantiate abstract class ChildFromPreprocess with abstract method RunPreprocessPipeline")

        class ChildFromPreprocess(Preprocess):
            def __init__(self):
                pass
            def PreprocessTestData(self):
                pass
            def PreprocessTrainData(self):
                pass
            def RunPreprocessPipeline(self):
                pass
        ChildFromPreprocess()




class TestModelInterface(unittest.TestCase):
    def test_Abstraction(self):

        class ChildFromModel(ModelInterface):
            def __init__(self):
                pass

        with self.assertRaises(TypeError) as message:
            ChildFromModel()
        self.assertEqual(str(message.exception), "Can't instantiate abstract class ChildFromModel with abstract methods Fit, GridSearch, Predict, model")

        class ChildFromModel(ModelInterface):
            def __init__(self):
                pass
            def Fit(self):
                pass
            def Predict(self):
                pass
                
        with self.assertRaises(TypeError) as message:
            ChildFromModel()
        self.assertEqual(str(message.exception), "Can't instantiate abstract class ChildFromModel with abstract methods GridSearch, model")

        class ChildFromModel(ModelInterface):
            def __init__(self):
                pass
            def Fit(self):
                pass
            def Predict(self):
                pass
            def GridSearch(self):
                pass
            def model(self):
                pass
        ChildFromModel()