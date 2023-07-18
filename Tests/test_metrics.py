import unittest
# import pytest

from Core.Metrics import Metrics

class TestMetrics(unittest.TestCase):
    def test_ClassificationMetrics(self):

        with self.assertRaises(ValueError) as message:
            Metrics().ClassificationMetrics("","")
        self.assertEqual(str(message.exception), "Expected array-like (array or non-string sequence), got ''")

        with self.assertRaises(ValueError) as message:
            Metrics().ClassificationMetrics([],[1])

    def test_F1(self):

        with self.assertRaises(ValueError) as message:
            Metrics().F1("","")
        self.assertEqual(str(message.exception), "Expected array-like (array or non-string sequence), got ''")

        with self.assertRaises(ValueError) as message:
            Metrics().F1([],[1])
        self.assertEqual(str(message.exception), "Found input variables with inconsistent numbers of samples: [0, 1]")

    def test_AUC(self):

        with self.assertRaises(ValueError) as message:
            Metrics().AUC("","")
        self.assertEqual(str(message.exception), "Expected array-like (array or non-string sequence), got ''")

        with self.assertRaises(ValueError) as message:
            Metrics().AUC([],[1])
        self.assertEqual(str(message.exception), "Found input variables with inconsistent numbers of samples: [0, 1]")

    def test_ConfusionMatrix(self):

        with self.assertRaises(ValueError) as message:
            Metrics().ConfusionMatrix("","")
        self.assertEqual(str(message.exception), "Expected array-like (array or non-string sequence), got ''")

        with self.assertRaises(ValueError) as message:
            Metrics().ConfusionMatrix([],[1])
        self.assertEqual(str(message.exception), "Found input variables with inconsistent numbers of samples: [0, 1]")