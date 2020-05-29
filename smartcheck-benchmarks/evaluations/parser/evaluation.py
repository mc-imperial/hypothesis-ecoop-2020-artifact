from evalcommon import support_executable
from subprocess import Popen, PIPE
import ast

SIZE = support_executable('parser', 'Size')


CACHE = {}


def size_haskell(output):
    try:
        return CACHE[output]
    except KeyError:
        pass
    proc = Popen([SIZE], stdin=PIPE, stdout=PIPE, encoding='utf-8')
    stdout, _ = proc.communicate(output)
    assert proc.returncode == 0
    return CACHE.setdefault(output, int(stdout.strip()))


def size_python(output):
    return size_haskell(ast.literal_eval(output))
