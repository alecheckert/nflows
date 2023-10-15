from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from .constants import DTYPE
from .model import Model


class Dataset(ABC):
    """Represents a training dataset for a normalizing flow, including
    a method for generating samples."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Total number of items in this dataset."""

    @abstractmethod
    def get_batch(self, n: int) -> np.ndarray:
        """Generate a sample of *n* data items."""



def adam(
    model: Model,
    dataset: Dataset,
    out_dir: str,
    learning_rate: float = 1e-3,
    epochs: int = 10000,
    batch_size: int = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-7,
    verbose: bool = True,
    save_mode: str = "best",
):
    """Optimize a model with the adaptive moment estimation
    (adam) algorithm.

    Parameters
    ----------
    model           :   model to train
    dataset         :   dataset on which to train
    out_dir         :   output directory (always created)
    learning_rate   :   damping factor for parameter updates
    epochs          :   total number of epochs to run
    batch_size      :   size of each minibatch. If float,
                        interpreted as the fraction of the
                        entire dataset to use in each minibatch;
                        if int, the size of the minibatch.
    beta1, beta2, epsilon: internal adam parameters (chosen
                        to match tensorflow/original paper)
    verbose         :   print progress at each epoch
    save_mode       :   one of:
                        "nosave": don't save anything
                        "best": save the best model so far
                        "all": save every model

    Returns
    -------

    """
    if isinstance(batch_size, float):
        batch_size = int(batch_size * dataset.size)
    elif not isinstance(batch_size, int):
        raise TypeError(f"unrecognized type for batch_size: {type(batch_size)}")
    assert batch_size <= dataset.size

    parameters = model.get_parameters(flat=True)
    mt = np.zeros(model.n_parameters, dtype=np.float64)
    vt = np.zeros(model.n_parameters, dtype=np.float64)
    epoch_records = []

    if os.path.isdir(out_dir):
        raise OSError(f"directory exists: {out_dir}")
    os.mkdir(out_dir)
    epoch_records_path = os.path.join(out_dir, "epoch_records.csv")
    model_path = os.path.join(out_dir, "%s_epoch{}.nflows" % os.path.basename(out_dir))
    best_loss = np.inf

    for epoch in range(epochs):
        batch = dataset.get_batch(batch_size)
        loss_value, _, gradient = model.backward(batch)
        mean_loss_value = loss_value.mean()
        mt = beta1 * mt + (1 - beta1) * gradient
        vt = beta2 * vt + (1 - beta2) * (gradient**2)
        mhat = mt / (1 - np.power(beta1, epoch+1))
        vhat = vt / (1 - np.power(beta2, epoch+1))
        parameters = parameters - learning_rate * mhat / (np.sqrt(vhat) + epsilon)
        model.set_parameters(parameters.astype(DTYPE))
        epoch_records.append(
            {
                "epoch": epoch,
                "mean_loss_value": mean_loss_value,
            }
        )

        # Show progress
        if verbose:
            print(f"Epoch {epoch}:\t{mean_loss_value:.3f}")
            # print(pd.DataFrame(epoch_records))

        # Save, if desired
        if save_mode == "all" or (save_mode == "best" and mean_loss_value < best_loss):
            pd.DataFrame(epoch_records).to_csv(epoch_records_path, index=False)
            model.save(model_path.format(epoch))

        # Update best loss
        best_loss = min(best_loss, mean_loss_value)

    return pd.DataFrame(epoch_records)
