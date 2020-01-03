import unittest
import pickle
from calibration_sklearn import Calibrator
class TestCalibration_Sklearn(unittest.TestCase):

    def test_fit_from_logits(self):
        logits = [[-2.75346231, -0.499726474, 2.48991966],
                  [3.08145404, -1.94772363, -0.449227452],
                  [1.58149862, -0.886176586, -0.384896457],
                  [-1.85654736, 2.10585546, -0.938781261]]
        y = [2, 0, 0, 1]
        calibrator = Calibrator(method="sigmoid")
        model = calibrator.fit(logits, y)
        self.assertIsNotNone(model)

    def test_predict_from_serialized_model(self):
        modelfp = "fixtures/calibration.pkl"
        with open(modelfp, "rb") as modelf:
            model = pickle.load(modelf)

        X = [[-1.85654736, 2.10585546, -0.938781261]]
        truth = [[0.27461498, 0.57483514, 0.15054989]]
        proba = model.predict_proba(X)
        pred = model.predict(X)
        self.assertEqual(proba.shape, (1,3))
        for idx, n in enumerate(truth[0]):
            self.assertAlmostEqual(proba[0][idx], n)
        self.assertEqual(pred, [1])
