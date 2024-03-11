from statpy.log import message
from statpy.database import custom_json as json
import zlib
from dataclasses import dataclass

@dataclass(frozen=True)
class Leaf:
    def __init__(self, mean, jks, sample, misc):
        self.mean = mean
        self.jks = jks
        self.sample = sample
        self.misc = misc 

    def _to_dict(self):
        return {"mean": self.mean, "jks": self.jks, "sample": self.sample, "misc": self.misc}
    
    def to_dict(self):
        data = self._to_dict()
        data["checksum"] = calculate_checksum(data)
        return data

    @classmethod
    def from_dict(cls, data):
        cksum = data.pop("checksum", None)
        tag = data["misc"].get("tag") if data["misc"] is not None else None
        if cksum:
            cksumcomp = calculate_checksum(data)
            if cksum != cksumcomp:
                raise Exception(f"{tag}: Data corrupted!")
        else:
            message(f"{tag}: Checksum missing. Data may be corrupt.") 
        if tag is not None: del data["misc"]["tag"]
        return cls(**data)

    
def calculate_checksum(data):
    data_json = json.dumps(data).encode('utf-8')  # Serialize as JSON first
    return zlib.crc32(data_json) & 0xFFFFFFFF  # Ensure unsigned 32-bit