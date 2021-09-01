import os


class Config(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
