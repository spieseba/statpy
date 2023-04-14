class Leaf:
    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)
    def to_dict(self):
        return self.__dict__
    def __init__(self, mean, jks, sample, info=None):
        self.mean = mean
        self.jks = jks
        self.sample = sample
        if info == None:
            self.info = {}

#class SampleLeaf():
#    @classmethod
#    def from_dict(cls, dct):
#        return cls(**dct)
#    def to_dict(self):
#        return self.__dict__
#    def __init__(self, sample, mean=None, jks=None, info=None):
#        self.sample = sample
#        self.mean = mean
#        self.jks = jks
#        if info == None:
#            self.info = {}