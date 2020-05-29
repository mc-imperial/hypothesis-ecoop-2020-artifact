import sys
from hypothesis import settings, Verbosity
from hypothesis.internal.conjecture.utils import calc_label_from_name
import click
import hypothesis.internal.conjecture.engine as eng
from hypothesis.internal.conjecture.data import ConjectureData, Status, StopTest
from random import Random
import random
import base64
import json
from contextlib import contextmanager
import signal
import hashlib
import traceback
import click
from hypothesis.internal.conjecture.shrinker import block_program
import time
import re


NUMERALS = re.compile("[0-9]+")


STEP_LABEL = calc_label_from_name("TSTL STEP")
DRAW_STEP = calc_label_from_name("TSTL STEP SELECTION")


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



CHOICE = calc_label_from_name("CHOICE")


def choice(data, n):
    n_bits = n.bit_length()
    while True:
        data.start_example(CHOICE)
        i = data.draw_bits(n_bits)
        if i >= n:
            data.stop_example(discard=True)
            continue
        data.stop_example()
        return i

FILTERED_ACTIONS = calc_label_from_name("FILTERED ACTIONS")


@click.group()
@click.option('--sut-file')
@click.option('--log-file')
@click.option('--seed')
@click.option('--n-steps')
@click.pass_context
def tstl(ctx, sut_file, log_file, seed, n_steps):
    eng.BUFFER_SIZE = 10 ** 6
    eng.MAX_SHRINKS = 10 ** 6

    with open(sut_file) as i:
        source = i.read()

    exec_globals = {}
    exec(source, exec_globals)

    sut_class = exec_globals["sut"]
    seed = int(seed)
    n_steps = int(n_steps)

    def run_tstl(data):
        data.extra_information.tstl_steps = getattr(data, 'tstl_steps', [])

        run_checks = False

        count = 0
        sut = sut_class()
        sut.restart()
        random.seed(0)

        actions = list(sut.actions())
        actions.sort(key=lambda t: t[0])

        while True:
            # We draw a value which in normal random mode will almost never be
            # zero, but when we are shrinking can be easily turned into a zero.
            data.start_example(STEP_LABEL)
            if count >= n_steps:
                should_continue = data.draw_bits(64, forced=0)
            else:
                should_continue = data.draw_bits(64)
            if not should_continue:
                data.stop_example()
                break

            count_discarded = data.draw_bits(64) == 0

            count += 1

            if count_discarded:
                data.start_example(CHOICE)
                i = choice(data, len(actions))
                _, guard, _ = actions[i]

                if not guard():
                    data.stop_example(discard=False)
                    continue
                else:
                    data.stop_example(discard=False)
            else:
                for _ in range(3):
                    data.start_example(CHOICE)
                    i = choice(data, len(actions))
                    _, guard, _ = actions[i]
                    succeeded = guard()
                    data.stop_example(discard=not succeeded)
                    if succeeded:
                        break
                else:
                    data.start_example(FILTERED_ACTIONS)
                    valid_actions = [i for i in range(len(actions)) if actions[i][1]()]
                    if not valid_actions:
                        data.mark_invalid()
                    j = choice(data, len(valid_actions))
                    i = valid_actions[j]
                    data.stop_example(discard=True)
                    data.draw_bits(len(actions).bit_length(), forced=i)

            name, guard, action = actions[i]

            assert guard()

            data.stop_example()

            data.extra_information.tstl_steps.append(name)

            failure = None

            try:
                with timeout(5):
                    action()
                    if run_checks and not sut.check():
                        failure = sut.failure()
            except TimeoutExpired:
                data.mark_invalid()
            # FIXME: Sympy specific hack
            except RecursionError:
                data.mark_invalid()
            except StopTest:
                raise
            except Exception:
                failure = sys.exc_info()

            if failure is not None:
                if not data.frozen:
                    data.draw_bits(64, forced=0)
                
                error_type, _, tb = failure

                origin = traceback.extract_tb(tb)[-1]
                filename = origin[0]
                lineno = origin[1]
                data.mark_interesting((error_type.__name__, filename, lineno))

    log = open(log_file, "w")
    def log_data(data):
        log_data = [
            hashlib.sha1(data.buffer).hexdigest()[:10],
            len(data.buffer),
            len(data.extra_information.tstl_steps),
            data.status.name,
        ]

        if data.status == Status.INTERESTING:
            log_data.extend([
                data.interesting_origin,
                base64.b64encode(data.buffer).decode('ascii'),
                data.extra_information.tstl_steps,
            ])
        log.write(json.dumps(log_data))
        log.write("\n")
        log.flush()

    ctx.ensure_object(dict)
    ctx.obj['run_tstl'] = run_tstl
    ctx.obj['log'] = log
    ctx.obj['log_data'] = log_data
    ctx.obj['seed'] = seed


@tstl.command()
@click.pass_context
def generate(ctx):
    random = Random(ctx.obj['seed'])
    log_data = ctx.obj['log_data']
    run_tstl = ctx.obj['run_tstl']

    while True:
        target = ConjectureData(
            prefix=b'', random=random, max_length=eng.BUFFER_SIZE,
        )
        try:
            run_tstl(target)
        except StopTest:
            pass

        log_data(target)

        if target.status == Status.INTERESTING:
            break

    ctx.obj['log'].close()


@tstl.command()
@click.argument('initial')
@click.option('--delete-only/--full', default=False)
@click.pass_context
def shrink(ctx, initial, delete_only):
    random = Random(ctx.obj['seed'])
    log_data = ctx.obj['log_data']
    log = ctx.obj['log']
    run_tstl = ctx.obj['run_tstl']

    runner = eng.ConjectureRunner(run_tstl,
        settings=settings(max_examples=10**6, database=None, verbosity=Verbosity.debug),
        random=random,
    )

    runner.debug_data = lambda data: None

    def debug(s):
        line = f"// {s.rstrip()}"
        log.write(line + "\n")

    runner.debug = debug

    original_test_function = runner.test_function

    def test_function(data):
        original_test_function(data)
        log_data(data)

    runner.test_function = test_function

    try:
        target = runner.cached_test_function(base64.b64decode(initial))

        if target.status != Status.INTERESTING:
            print(f"Cannot shrink target with status {target.status.name}", file=sys.stderr)
            sys.exit(1)

        shrinker = runner.new_shrinker(
            target, lambda d: d.status == Status.INTERESTING and d.interesting_origin == target.interesting_origin
        )

        if delete_only:
            shrinker.fixate_shrink_passes(["adaptive_example_deletion"])
        else:
            shrinker.shrink()
    except eng.RunIsComplete:
        pass
    except Exception:
        debug("ERROR RUNNING TEST")
        for l in traceback.format_exc().splitlines():
            debug(l)

    for v in runner.interesting_examples.values():
        log_data(v)

    log.close()


if __name__ == '__main__':
    tstl()
