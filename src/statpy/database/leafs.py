import hashlib

class Leaf:
    def __init__(self, mean, jks, sample, misc=None):
        self.mean = mean
        self.jks = jks
        self.sample = sample
        self.misc = misc

    @classmethod
    def from_dict(cls, dct):
        # verify checksum!
        return cls(**dct)
    
    def to_dict(self):
        data = self._to_dict()
        #data['checksum'] = self._calculate_checksum(data)
        return data
    
    def _to_dict(self):
        return self.__dict__
     
    def _calculate_checksum(self, data):
        hash_function = hashlib.sha1()  
        hash_function.update(str(data).encode('utf-8'))  
        return hash_function.hexdigest() 