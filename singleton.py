# REF: http://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons
def singleton(cls):
    obj = cls()
    # Always return the same object
    cls.__new__ = staticmethod(lambda cls: obj)
    # Disable __init__
    try:
        del cls.__init__
    except AttributeError:
        pass
    return cls
