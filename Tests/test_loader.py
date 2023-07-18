import pytest
import unittest
from unittest.mock import patch
from DataLoader.SupportedScalers import SupportedScalers
from sklearn.preprocessing import StandardScaler
import pprint
import sys

from DataLoader.Loader import TitanicData

class TestMetrics(unittest.TestCase):

    def test_Init(self):
        titanicData = TitanicData()

        self.assertEqual(
            titanicData.train_x.shape[1],
            titanicData.test_x.shape[1]
        )

        self.assertIsInstance(
            titanicData.scaler,SupportedScalers
        )

    def test_RunPreprocessPipeline(self):

        titanicData = TitanicData()
        train_x, train_Y, test_x, test_Y = titanicData.RunPreprocessPipeline()
        self.assertEqual(
            train_x.shape[1],
            test_x.shape[1]
        )