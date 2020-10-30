import re
import ast


def size_python(value):
    value = ast.literal_eval(value)

    def w(x):
        if isinstance(x, tuple):
            return w(x[1]) + w(x[2]) + 1
        else:
            return 1
    return w(value)


CONSTRUCTOR = re.compile('C|Add|Div')


def size_haskell(output):
    return len(CONSTRUCTOR.findall(output))
