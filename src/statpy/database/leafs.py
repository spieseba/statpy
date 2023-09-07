import numpy as np

class Leaf:
    def __init__(self, mean, jks, sample, misc=None):
        self.mean = mean
        self.jks = jks
        self.sample = sample
        self.misc = misc

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)
    
    def to_dict(self):
        return self.__dict__
