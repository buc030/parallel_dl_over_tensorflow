
import os

def check_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def printInfo(msg):
    pass
