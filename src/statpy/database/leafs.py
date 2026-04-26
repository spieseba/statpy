from statpy.log import message
from statpy.database import custom_json as json
import zlib
from frozendict import frozendict

class Leaf:
    def __init__(self, mean, jks, sample, bss=None, misc=None, weights_tag=None):
        self._mean = mean
        self._jks = frozendict(jks) if jks is not None else jks
        self._sample = frozendict(sample) if sample is not None else sample
        self._bss = bss
        self._misc = frozendict(misc) if misc is not None else misc
        self._weights_tag = weights_tag

    @property
    def mean(self):
        return self._mean
    @property
    def jks(self):
        return self._jks
    @property
    def sample(self):
        return self._sample
    @property
    def bss(self):
        return self._bss
    @property
    def misc(self):
        return self._misc
    @property
    def weights_tag(self):
        return self._weights_tag

    def _to_dict(self):
        return {"mean": self.mean, "jks": self.jks, "sample": self.sample, "bss": self.bss, "misc": self.misc, "weights_tag": self.weights_tag}

    def to_dict(self):
        data = self._to_dict()
        data["checksum"] = calculate_checksum(data)
        return data

    @classmethod
    def from_dict(cls, data):
        cksum = data.pop("checksum", None)
        if cksum is None:
            message("Leaf checksum missing; data may be corrupt.")
        elif cksum != calculate_checksum(data):
            raise Exception("Leaf checksum mismatch; data corrupted.")
        return cls(**data)
    
def calculate_checksum(data):
    data_json = json.dumps(data).encode('utf-8')  # Serialize as JSON first
    return zlib.crc32(data_json) & 0xFFFFFFFF  # Ensure unsigned 32-bit