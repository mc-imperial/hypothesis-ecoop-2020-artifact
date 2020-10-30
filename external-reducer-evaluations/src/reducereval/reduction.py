from enum import Enum
import attr
import time
from time import monotonic as monotonic_time
from reducereval import WORKING, ROOT
from reducereval.filesystem import mkdirp, claim_lock, release_lock
import os
import tempfile
import signal
import shutil
from subprocess import Popen
import subprocess
from contextlib import contextmanager
import random
import struct


class Classification(Enum):
    INVALIDCHEAP = 0
    INVALIDEXPENSIVE = 1
    VALID = 2
    INTERESTING = 3
    TIMEOUT = 4
    SLIPPAGE = 5


@attr.s()
class ExampleStatistics(object):
    count = attr.ib(default=0)
    total_size = attr.ib(default=0)
    runtime = attr.ib(default=0.0)
    values = attr.ib(init=False, repr=False, default=attr.Factory(list))

    def record(self, size, runtime):
        self.count += 1
        self.total_size += size
        self.runtime += runtime
        self.values.append((size, runtime))


@attr.s()
class ReducerStats(object):
    best_example = attr.ib(default=None)
    start_time = attr.ib(default=None)
    end_time = attr.ib(default=None)
    statistics = attr.ib(
        default=attr.Factory(
            lambda: {c.name: ExampleStatistics() for c in Classification}
        )
    )

    def start(self):
        self.start_time = monotonic_time()

    def finish(self):
        self.end_time = monotonic_time()


def tracking_predicate(predicate):
    assert predicate is not None
    result = ReducerStats()

    def tracking_predicate(value):
        start = monotonic_time()
        try:
            classification = predicate(value)
        except TimeoutExpired:
            classification = Classification.TIMEOUT
        end = monotonic_time()
        result.statistics[classification.name].record(
            runtime=end - start, size=len(value)
        )
        if classification == Classification.INTERESTING and (
            result.best_example is None or len(value) <= len(result.best_example)
        ):
            result.best_example = value
        return classification

    return result, tracking_predicate


CALLBACK_EXE = os.path.join(WORKING, "tools", "callback")
CALLBACK_SRC = os.path.join(ROOT, "tools", "callback.c")

BUILT_CALLBACK = False


def delete_older_target(source, target):
    if not os.path.exists(target):
        return
    if claim_lock(target):
        try:
            if (
                os.path.exists(target)
                and os.stat(source).st_mtime > os.stat(target).st_mtime
            ):
                os.unlink(target)
        finally:
            release_lock(target)


def atomic_compile(source, target):
    mkdirp(os.path.dirname(target))
    delete_older_target(source, target)
    if os.path.exists(target):
        return
    tmp_target = target + f"-{random.getrandbits(64)}.tmp"
    try:
        subprocess.check_call(
            ["gcc", "-Wall", "-Werror", "-O2", source, "-o", tmp_target]
        )
        os.rename(tmp_target, target)
    except FileExistsError:
        pass
    finally:
        try:
            os.unlink(tmp_target)
        except FileNotFoundError:
            pass


def callback_executable():
    global BUILT_CALLBACK
    if not BUILT_CALLBACK:
        atomic_compile(CALLBACK_SRC, CALLBACK_EXE)
        BUILT_CALLBACK = True
    return CALLBACK_EXE


class TimeoutExpired(Exception):
    pass


@contextmanager
def timeout(seconds):
    def raiseexc(signum, frame):
        raise TimeoutExpired()

    prev = None
    try:
        try:
            prev = signal.signal(signal.SIGALRM, raiseexc)
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        if prev is not None:
            signal.signal(signal.SIGALRM, prev)


class FileBasedReducer(object):
    def reduction_command(self, filepath):
        raise NotImplementedError()

    def gather(self):
        raise NotImplementedError()

    def run(self, initial, predicate):
        responses = None
        prev_handler = None
        tmpdir = tempfile.mkdtemp("reducer")

        local_tmpdir = os.path.join(tmpdir, "moretmp")
        os.mkdir(local_tmpdir)

        prevdir = os.getcwd()
        try:
            os.chdir(tmpdir)
            initial_file = os.path.join(tmpdir, "target")
            with open(initial_file, "w") as o:
                o.write(initial)
            cmd = self.reduction_command(initial_file)

            filenames = os.path.join(tmpdir, "filenames")
            results = os.path.join(tmpdir, "exitcodes")

            os.mkfifo(filenames)
            os.mkfifo(results)

            env = dict(os.environ)
            env["CALLBACKFILESFIFO"] = filenames
            env["CALLBACKRESULTSFIFO"] = results
            env["TMPDIR"] = local_tmpdir

            process = Popen(cmd, cwd=tmpdir, env=env)

            with open(filenames, "rb") as requests:
                buf = bytearray()
                while True:
                    try:
                        with timeout(0.1):
                            c = requests.read(1)
                    except TimeoutExpired:
                        c = b""

                    if not c:
                        process.poll()
                        if process.returncode is None:
                            continue
                        else:
                            break
                    if c != b"\n":
                        buf.extend(c)
                        continue
                    callback, filename = buf.decode("utf-8").split(" ", 1)
                    buf.clear()
                    with open(filename) as i:
                        value = i.read()
                    if responses is None:
                        responses = open(results, "wb", 0)

                    try:
                        with timeout(30):
                            classification = predicate(value)
                    except TimeoutExpired:
                        classification = None

                    resp = struct.pack(
                        ">Ic",
                        int(callback),
                        b"\0"
                        if classification == Classification.INTERESTING
                        else b"\1",
                    )
                    assert len(resp) == 5
                    responses.write(resp)
                    responses.flush()
            result = self.gather()
            assert predicate(result)
            return result
        finally:
            os.chdir(prevdir)
            if responses is not None:
                responses.close()
            if prev_handler is not None:
                signal.signal(signal.SIGCHLD, prev_handler)
            try_delete_dir(tmpdir)


def try_delete_dir(tmpdir):
    for _ in range(3):
        try:
            shutil.rmtree(tmpdir)
            return
        except OSError:
            pass
        time.sleep(1)


class Picire(FileBasedReducer):
    def __init__(self, atom="both"):
        self.atom = atom

    def reduction_command(self, source):
        return [
            "picire",
            "--input",
            source,
            "--test",
            callback_executable(),
            f"--atom={self.atom}",
            "--out",
            "results",
        ]

    def gather(self):
        with open("results/target") as i:
            return i.read()


def run_picire(initial, predicate):
    return Picire().run(initial, predicate)


SCRIPT_TEMPLATE = """
#!/usr/bin/env sh
 
if %(callback)s %(name)s ; then
    exit 0
else
    exit 1
fi
"""


class CReduce(FileBasedReducer):
    def __init__(self, not_c):
        self.not_c = not_c

    def gather(self):
        with open("target") as i:
            return i.read()

    def reduction_command(self, source):
        name = os.path.basename(source)
        script = os.path.abspath("test.sh")
        with open(script, "w") as o:
            o.write(
                SCRIPT_TEMPLATE
                % {"callback": callback_executable(), "name": os.path.basename(source)}
            )
        os.chmod(script, 0o775)
        commands = ["creduce", script, source, "--n", "1", "--timeout", "1000000"]
        if self.not_c:
            commands.append("--not-c")
        return commands


def run_creduce(initial, predicate, not_c=False):
    return CReduce(not_c=not_c).run(initial, predicate)
