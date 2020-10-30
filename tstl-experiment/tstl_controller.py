import attr
import traceback
import sys
import random
import json
import signal
import time
from contextlib import contextmanager


class TimeoutExpired(Exception):
    pass


@contextmanager
def timeout(seconds):
    def raiseexc(signum, frame):
        raise TimeoutExpired()

    start = time.monotonic()
    prev = None
    try:
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            prev = signal.signal(signal.SIGALRM, raiseexc)
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        if prev is not None:
            signal.signal(signal.SIGALRM, prev)
    end = time.monotonic()
    if end - start >= seconds:
        raise TimeoutExpired()

@attr.s()
class TSTLResult(object):
    source_program = attr.ib()
    enabled = attr.ib(default=attr.Factory(list))
    read = attr.ib(default=0)
    outcome = attr.ib(default=None)


def run_tstl(program):
    result = TSTLResult(source_program=program)

    from sut import sut as sut_class

    sut = sut_class()

    actions = list(sut.actions())

    actions_by_name = {s: i for i, (s, _, _) in enumerate(actions)}

    sut.restart()

    try:
        for action in result.source_program:
            i = actions_by_name[action]
            result.read += 1

            name, guard, act = sut.actions()[i]

            with timeout(5):
                if not guard():
                    result.enabled.append(False)
                else:
                    result.enabled.append(True)
                    act()
    except Exception:
        error_type, _, tb = sys.exc_info()
        origin = traceback.extract_tb(tb)[-1]
        filename = origin[0]
        lineno = origin[1]
        result.outcome = (error_type.__name__, filename, lineno)
    return result


if __name__ == '__main__':
    program, seed = json.loads(sys.stdin.read())
    random.seed(seed)
    result = run_tstl(program)
    print(json.dumps(
        attr.asdict(result)
    ))
