import sys
import click
from random import Random
import random
import base64
import json
from contextlib import contextmanager
import signal
import hashlib
import traceback
import click
import time
import re



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


def run_tstl(sut_class, program):
    sut = sut_class()
    sut.restart()

    steps = []

    actions = {t[0]: t for t in sut.actions()}

    failure = None

    for act in program:
        name, guard, action = actions[act]

        if not guard():
            continue
        steps.append(name)
        try:
            with timeout(5):
                action()
        except Exception:
            error_type, _, tb = sys.exc_info()

            origin = traceback.extract_tb(tb)[-1]
            filename = origin[0]
            lineno = origin[1]
            failure = (error_type.__name__, filename, lineno)
    return (tuple(steps), failure)


@click.command()
@click.option("--sut-file")
@click.option("--log-file")
@click.option("--program-file")
def main(sut_file, log_file, program_file):
    with open(program_file) as i:
        program = tuple(l.rstrip("\n") for l in i)

    with open(sut_file) as i:
        sut_source = i.read()

    exec_globals = {}
    exec(sut_source, exec_globals)

    sut_class = exec_globals["sut"]

    seen = {()}
    call_count = 0

    program, target = run_tstl(sut_class, program)

    assert target[0] != 'TimeoutExpired'

    k = len(program)
    prev = program
    while prev != program or k > 1:
        if prev == program:
            k //= 2
        assert k > 0
        prev = program
        i = 0
        while i + k <= len(program):
            attempt = program[:i] + program[i + k:]
            assert len(attempt) < len(program)
            if attempt not in seen:
                call_count += 1
                seen.add(attempt)
                pruned, error = run_tstl(sut_class, attempt)
                seen.add(pruned)
                if error == target:
                    program = pruned
            i += k

    with open(log_file, "w") as o:
        print(json.dumps({'result': program, 'calls': call_count}), file=o)


if __name__ == '__main__':
    main()
