from statpy.log import message
from statpy.database import custom_json as json
import zlib
from frozendict import frozendict

class Leaf:
    def __init__(self, mean, jks, sample, misc):
        self._mean = mean
        self._jks = frozendict(jks) if jks is not None else jks
        self._sample = frozendict(sample) if sample is not None else sample
        self._misc = frozendict(misc) if misc is not None else misc

    @property
    def mean(self):
        return self._mean
    @property
    def jks(self):
        return self._jks #.copy() if self._jks is not None else None
    @property
    def sample(self):
        return self._sample
    @property
    def misc(self):
        return self._misc
     
    def _to_dict(self):
        return {"mean": self.mean, "jks": self.jks, "sample": self.sample, "misc": self.misc}
    
    def to_dict(self):
        data = self._to_dict()
        data["checksum"] = calculate_checksum(data)
        return data

    @classmethod
    def from_dict(cls, data):
        cksum = data.pop("checksum", None)
        if cksum:
            cksumcomp = calculate_checksum(data)
            if cksum != cksumcomp:
                raise Exception(f"Data corrupted!")
        else:
            message(f"Checksum missing. Data may be corrupt.") 
        return cls(**data)

    
def calculate_checksum(data):
    data_json = json.dumps(data).encode('utf-8')  # Serialize as JSON first
    return zlib.crc32(data_json) & 0xFFFFFFFF  # Ensure unsigned 32-bit