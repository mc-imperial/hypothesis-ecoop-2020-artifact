import re


NUMBERS = re.compile('-?[0-9]+')


def size(output):
    return len(NUMBERS.findall(output))
