from abc import ABC


class BaseAttribute(ABC):
    """Base class for estimating a feature importance."""

    @abstractmethod
    def estimate(self, data_batch, data_samples):
        pass
