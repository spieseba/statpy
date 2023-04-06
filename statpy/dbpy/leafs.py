class Leaf:
    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)
    def to_dict(self):
        return self.__dict__
    def __init__(self, mean, jks):
        self.mean = mean
        self.jks = jks

class SampleLeaf():
    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)
    def to_dict(self):
        return self.__dict__
    def __init__(self, sample, mean=None, jks=None):
        self.sample = sample
        self.mean = mean
        self.jks = jks