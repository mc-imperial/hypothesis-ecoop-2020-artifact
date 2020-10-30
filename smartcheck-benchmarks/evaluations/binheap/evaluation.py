import re
import ast


def size_python(value):
    value = ast.literal_eval(value)

    def w(x):
        if x is None:
            return 1
        return w(x[1]) + w(x[2]) + 1
    return w(value)


CONSTRUCTOR = re.compile('Node|Empty')


def size_haskell(output):
    return len(CONSTRUCTOR.findall(output))
