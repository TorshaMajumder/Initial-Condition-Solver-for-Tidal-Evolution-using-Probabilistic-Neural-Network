import sys


class Logger(object):
    def __init__(self, path_to_store=None):
        self.terminal = sys.stdout
        self.log = open(f"{path_to_store}/poet_solver.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def start(path):
    sys.stdout = Logger(path_to_store=path)


def stop():
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal