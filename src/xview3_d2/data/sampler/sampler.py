import itertools
import logging
from typing import Optional, Sequence

import torch
from detectron2.data.samplers import TrainingSampler
from detectron2.utils import comm

logger = logging.getLogger(__name__)

__all__ = ["WeightedRandomTrainingSampler"]


class WeightedRandomTrainingSampler(TrainingSampler):
    """
    Combined Detectron2 infinite stream ``TrainingSampler`` with PyTorch ``WeightedRandomSampler``.

    Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    """

    def __init__(
        self,
        size: int,
        weights: Optional[Sequence[float]] = None,
        shuffle: bool = True,
        replacement: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            weights (sequence)   : a sequence of weights, not necessary summing up to one
            replacement (bool): if ``True``, samples are drawn with replacement.
                        If not, they are drawn without replacement, which means that when a
                        sample index is drawn for a row, it cannot be drawn again for that row.
            seed (int): the initial seed of the shuffle. Must be the same
                        across all workers. If None, will use a random seed shared
                        among workers (require synchronization among all workers).
        """
        if not isinstance(size, int):
            raise TypeError(
                f"WeightedRandomTrainingSampler(size=) expects an int. Got type {type(size)}."
            )
        if size <= 0:
            raise ValueError(
                f"WeightedRandomTrainingSampler(size=) expects a positive int. Got {size}."
            )
        self._size = size
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        self._shuffle = shuffle
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        self.weights = (
            torch.as_tensor(weights, dtype=torch.double)
            if weights is not None
            else None
        )
        self.replacement = replacement

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        generator = torch.Generator()
        generator.manual_seed(self._seed)

        while True:
            if self.weights is None:
                indices = torch.arange(self._size)
            else:
                indices = torch.multinomial(
                    self.weights,
                    num_samples=self._size,
                    replacement=self.replacement,
                    generator=generator,
                )

            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=generator).tolist()
                yield from indices[randperm].tolist()
            else:
                yield from indices.tolist()
