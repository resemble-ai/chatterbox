class AttrDict(dict):
    """Dictionary that allows attribute-style access to keys."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
