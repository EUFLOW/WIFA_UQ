""" Base class for a postprocessing block. """

from abc import ABC, abstractmethod
from dataclasses import dataclass

class PostProcesser(ABC):
    """
    Base class for a postprocessing block.
    """
    
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass
    