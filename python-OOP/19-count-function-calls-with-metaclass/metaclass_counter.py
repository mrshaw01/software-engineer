class FuncCallCounter(type):
    """A Metaclass which decorates all methods of the
    subclass using call_counter as the decorator.
    """

    @staticmethod
    def call_counter(func):
        def helper(*args, **kwargs):
            helper.calls += 1
            return func(*args, **kwargs)

        helper.calls = 0
        helper.__name__ = func.__name__
        return helper

    def __new__(cls, clsname, superclasses, attributedict):
        for attr in attributedict:
            if callable(attributedict[attr]) and not attr.startswith("__"):
                attributedict[attr] = cls.call_counter(attributedict[attr])
        return super().__new__(cls, clsname, superclasses, attributedict)
