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
