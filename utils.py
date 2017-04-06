
import os

def check_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def printInfo(msg):
    pass


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


