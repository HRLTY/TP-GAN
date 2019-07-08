from singleton import singleton;

@singleton
class ConfigFactory(object):
    def __init__(self):
        # super(type(self), self).__init(*args, **kwargs);
        self._flags = None;

    @property
    def flags(self):          return self._flags;

    @flags.setter
    def flags(self, value):   self._flags = value;
