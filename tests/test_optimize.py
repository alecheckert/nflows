import os
import unittest
import tempfile
import numpy as np
from glob import glob

from nflows.constants import SUFFIX
from nflows.flows import AffineFlow
from nflows.model import Model
from nflows.optimize import Dataset, adam


class TestAdam(unittest.TestCase):
    """Mostly a does-it-run test."""
    def test_adam(self):
        class GaussianDataset(Dataset):
            def __init__(self, size, mean, cov):
                self.X = np.random.multivariate_normal(mean=mean, cov=cov, size=size)

            @property
            def size(self) -> int:
                return self.X.shape[0]

            def get_batch(self, n: int) -> np.ndarray:
                indices = np.random.choice(np.arange(self.size), size=n, replace=False)
                return self.X[indices, :].copy()

        mean = np.array([1.0, -0.5])
        cov = np.array([[2.0, -0.5], [-0.5, 1.2]])
        size = 100
        model = Model([AffineFlow(2)])
        dataset = GaussianDataset(size, mean, cov)
        tempdir = tempfile.TemporaryDirectory()
        out_dir = os.path.join(tempdir.name, "testmodel")
        assert not os.path.isdir(out_dir)
        adam(
            model,
            dataset,
            out_dir,
            epochs=10,
            batch_size=10,
            verbose=False,
            save_mode="best",
        )
        assert os.path.isdir(out_dir)

        # Saves models that can be loaded
        model_paths = sorted(glob(os.path.join(out_dir, f"*{SUFFIX}")))
        assert len(model_paths) > 0
        model2 = Model.load(model_paths[-1])
        assert len(model2.flows) == len(model.flows)
