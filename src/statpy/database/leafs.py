import numpy as np

class Leaf:
    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)
    def to_dict(self):
        return self.__dict__
    def __init__(self, mean, jks, sample, misc=None):
        self.mean = mean
        self.jks = jks
        self.sample = sample
        self.misc = misc