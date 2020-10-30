import sys
from reducereval import ROOT, CORPORA, WORKING
import os

sys.path.append(os.path.join(ROOT, "hypothesis-csmith"))

# Use a local checkout of Hypothesis if available.
HYPOTHESIS_SOURCE = os.path.join(ROOT, "hypothesis", "hypothesis-python", "src")
if os.path.exists(HYPOTHESIS_SOURCE):
    sys.path.insert(0, HYPOTHESIS_SOURCE)

from hypothesis.core import encode_failure
import base64
import click
from reducereval.experiments import EXPERIMENTS, Experiment
from hypothesis.internal.conjecture.data import ConjectureData, Status, StopTest
from hypothesis.internal.conjecture.junkdrawer import uniform
import glob
import random
import json
from multiprocessing import cpu_count
import time
from contextlib import contextmanager

import hypothesis.internal.conjecture.engine as eng
from hypothesis import settings, Verbosity, HealthCheck
from hypothesis.errors import Flaky
import hashlib
from reducereval.reduction import tracking_predicate, Classification, CReduce, Picire, ExampleStatistics
from random import Random
import traceback
import shutil
import subprocess
import attr
from datetime import datetime
import signal
import linecache
import tracemalloc
from reducereval.filesystem import mkdirp, claim_lock, release_lock
import tempfile
from hypothesis.errors import UnsatisfiedAssumption
from collections import Counter, defaultdict
import heapq
from tqdm import tqdm


trace_memory_usage = False


# Put Hypothesis into infinite shrinking mode - its default behaviour is
# optimised for very short running tests.
eng.MAX_SHRINKS = float("inf")

BUFFER_SIZE = 10 ** 10


@click.group()
def main():
    pass


def atomic_create_file(target, contents):
    tmp_target = target + ".tmp-" + str(random.getrandbits(64))

    try:
        with open(tmp_target, "xb") as o:
            o.write(contents)

        try:
            os.rename(tmp_target, target)
        except FileExistsError:
            pass

    finally:
        try:
            os.unlink(tmp_target)
        except FileNotFoundError:
            pass


def unpacked_corpus_dir(name):
    return os.path.join(WORKING, "unpacked-corpora", name)


def corpus_for_experiment(name):
    destination = unpacked_corpus_dir(name)
    # TODO: Deal with corpus update
    if not os.path.exists(destination):
        tmpdir = destination + f".tmp-{random.getrandbits(64)}"
        try:
            os.makedirs(tmpdir)
            subprocess.check_call(
                ["tar", "--extract", "-f", os.path.join(CORPORA, name + ".tar")],
                cwd=tmpdir,
            )
            unpacked, = os.listdir(tmpdir)
            assert unpacked == name
            try:
                os.rename(os.path.join(tmpdir, name), destination)
            except FileExistsError:
                pass
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    assert os.path.exists(destination)
    return os.listdir(destination)


@attr.s()
class Codec(object):
    name = attr.ib()
    encode = attr.ib(repr=False)
    decode = attr.ib(repr=False)


UTF8 = Codec(
    name="UTF8", encode=lambda x: x.encode("utf-8"), decode=lambda x: x.decode("utf-8")
)


def raw_buffer(experiment, name):
    if isinstance(experiment, Experiment):
        experiment = experiment.name
    with open(os.path.join(unpacked_corpus_dir(experiment), name), "rb") as i:
        return i.read()


def file_size(experiment, name):
    if isinstance(experiment, Experiment):
        experiment = experiment.name
    return os.stat(os.path.join(unpacked_corpus_dir(experiment), name)).st_size


class Locked(Exception):
    pass


def cached_transformation(codec):
    def accept(fn):
        cachedir = os.path.join(WORKING, "cache", fn.__name__)
        mkdirp(cachedir)

        def cached_fn(*args, skip_locked=False, ignore_cache=False):
            args = [a.name if isinstance(a, Experiment) else a for a in args]
            keyparts = [getattr(f, '__name__', f) for f in args]
            cachefile = os.path.join(cachedir, "::".join(keyparts))
            start = time.monotonic()
            while not os.path.exists(cachefile) or ignore_cache:
                locked = False
                try:
                    locked = claim_lock(cachefile)
                    if not locked and skip_locked:
                        raise Locked()
                    if locked or time.monotonic() >= start + 30:
                        sys.stderr.write(
                            f"{fn.__name__}({', '.join(map(repr, keyparts))})\n"
                        )
                        result = fn(*args)
                        atomic_create_file(cachefile, codec.encode(result))
                        return result
                finally:
                    if locked:
                        release_lock(cachefile)
                time.sleep(random.random())
            with open(cachefile, "rb") as i:
                return codec.decode(i.read())

        cached_fn.__name__ = fn.__name__
        cached_fn.__qualname__ = fn.__qualname__
        return cached_fn

    return accept


def buffer_to_value(experiment, buffer):
    return ConjectureData.for_buffer(buffer).draw(experiment.generator)


@cached_transformation(UTF8)
def generate(experiment, name):
    return buffer_to_value(EXPERIMENTS[experiment], raw_buffer(experiment, name))


JSON = Codec(
    "JSON",
    encode=lambda value: json.dumps(value, sort_keys=True, indent=4).encode("utf-8"),
    decode=lambda binary: json.loads(binary.decode("utf-8")),
)


@cached_transformation(JSON)
def info(experiment, name):
    return EXPERIMENTS[experiment].calculate_info(generate(experiment, name))


def display_top(snapshot, key_type="lineno", limit=10):
    """From https://docs.python.org/3/library/tracemalloc.html#pretty-top,
    heavily modified.
    """
    snapshot = snapshot.filter_traces((tracemalloc.Filter(True, "*hypothesis*"),))
    top_stats = snapshot.statistics(key_type)

    print("Top %s allocating lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print(
            "  #%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
        )

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
    print()


@cached_transformation(JSON)
def reduce_with_hypothesis(experiment, name):
    return reduce_with_hypothesis_base(experiment, name, False)

@cached_transformation(JSON)
def reduce_badly_with_hypothesis(experiment, name):
    return reduce_with_hypothesis_base(experiment, name, True)


def reduce_with_hypothesis_base(experiment, name, suppress_intervals):
    experiment = EXPERIMENTS[experiment]

    base_predicate = experiment.calculate_error_predicate(info(experiment, name))
    generated = generate(experiment, name)
    classified = base_predicate(generated)

    assert classified == Classification.INTERESTING, (classified, base_predicate)

    results, predicate = tracking_predicate(base_predicate)

    generation_stats = {
        c: ExampleStatistics()
        for c in Status
    }

    input_to_outputs = []

    def test_function(data):
        if suppress_intervals:
            start = data.start_example
            stop = data.stop_example

            def start_example(label):
                if data.depth >= 0:
                    data.depth += 1
                    if data.depth > data.max_depth:
                        data.max_depth = data.depth
                else:
                    start(label)

            def stop_example(discard=False):
                if data.depth >= 1:
                    data.depth -= 1
                else:
                    stop(discard)
            data.start_example = start_example
            data.stop_example = stop_example

        generation_start = time.monotonic()
        try:
            try:
                source = data.draw(experiment.generator)
            except UnsatisfiedAssumption:
                data.mark_invalid()
            finally:
                generation_time = time.monotonic() - generation_start
            result = predicate(source)
            input_to_outputs.append((encode_failure(data.buffer).decode('ascii'), source, result.name))
            if trace_memory_usage:
                display_top(tracemalloc.take_snapshot())
            if result == Classification.INTERESTING:
                data.mark_interesting()
            elif result in (Classification.INVALIDCHEAP, Classification.INVALIDEXPENSIVE):
                data.mark_invalid()
        finally:
            generation_stats[data.status].record(size=len(data.buffer), runtime=generation_time)
 

    buffer = raw_buffer(experiment, name)

    runner = eng.ConjectureRunner(
        test_function,
        settings=settings(
            database=None,
            max_examples=1,
            suppress_health_check=HealthCheck.all(),
            deadline=None,
            verbosity=Verbosity.debug,
            buffer_size=BUFFER_SIZE,
        ),
        random=Random(int.from_bytes(hashlib.sha1(buffer).digest(), "big")),
    )

    def debug_data(data):
        runner.debug(
            f"DATA {hashlib.sha1(data.buffer).hexdigest()[:8]}: {len(data.buffer)} bytes, {data.status.name}"
        )

    runner.debug_data = debug_data

    runner.cached_test_function(buffer)
    assert runner.interesting_examples
    results.start()
    runner.shrink_interesting_examples()
    results.finish()
    v, = runner.interesting_examples.values()
    return {
        "final": {
            "buffer": base64.b64encode(v.buffer).decode("ascii"),
            "generated": buffer_to_value(experiment, v.buffer),
        },
        "reductionstats": attr.asdict(results),
        "input_to_outputs": input_to_outputs,
        "generationstats": {k.name: attr.asdict(v) for k, v in generation_stats.items()},
    }


def run_reducer(experiment, name, reducer):
    experiment = EXPERIMENTS[experiment]
    base_predicate = experiment.calculate_error_predicate(info(experiment, name), check_validity=True)
    print(f"Reducing for {base_predicate.__name__}")
    generated = generate(experiment, name)

    return run_reducer_base(generated, base_predicate, reducer)


def run_reducer_base(source, base_predicate, reducer):
    classified = base_predicate(source)

    assert classified == Classification.INTERESTING, (classified, base_predicate)

    results, predicate = tracking_predicate(base_predicate)

    results.start()
    result = reducer.run(source, predicate)
    results.finish()

    return {
        "bug": base_predicate.__name__,
        "reductionstats": attr.asdict(results),
        "final": result,
    }


@cached_transformation(JSON)
def creduce_on_hypothesis_output(experiment, name):
    experiment = EXPERIMENTS[experiment]
    base_predicate = experiment.calculate_error_predicate(info(experiment, name), check_validity=True)
    source = reduce_with_hypothesis(experiment, name)["final"]["generated"]
    return run_reducer_base(
        source, base_predicate, CReduce(not_c=experiment.name != "csmith")
    )


@cached_transformation(JSON)
def reduce_with_picire(experiment, name):
    return run_reducer(experiment, name, Picire())


@cached_transformation(JSON)
def reduce_with_creduce(experiment, name):
    return run_reducer(experiment, name, CReduce(not_c=experiment != "csmith"))


def fork_pool(n_workers, run):
    forked = set()
    crashes = 0
    give_up = False

    try:
        while n_workers > 0:
            while len(forked) < n_workers:
                seed = random.getrandbits(128)
                pid = os.fork()
                if pid == 0:
                    try:
                        random.seed(seed ^ os.getpid())
                        run()
                        os._exit(0)
                    except:
                        traceback.print_exc()
                        os._exit(1)
                else:
                    print(f"Spawning worker process {pid}")
                    forked.add(pid)
            pid, result = os.wait()
            forked.remove(pid)
            if result == 0:
                print(f"Worker {pid} exited normally.")
                n_workers -= 1
            elif give_up:
                n_workers -= 1
            else:
                crashes += 1
                if crashes >= 100:
                    print("Too many crashes. Will stop spawning workers now.")
                    give_up = True
                else:
                    print(f"Worker {pid} crashed, restarting.")
    finally:
        if forked:
            assert os.getpid() == original_pid
            print(f"Killing workers {', '.join(map(str, forked))}")
            for child in forked:
                os.kill(child, signal.SIGINT)

            i = 0
            while i < 2 and forked:
                time.sleep(0.5)

                for pid in list(forked):
                    rpid, result = os.waitpid(pid, os.WNOHANG)
                    if pid == rpid:
                        forked.remove(pid)
                i += 1
            for child in forked:
                os.kill(child, signal.SIGKILL)


original_pid = os.getpid()


def build_all():
    if os.getpid() != original_pid:
        logdir = os.path.join(WORKING, "logs")
        mkdirp(logdir)
        counter = 0
        extra = ""
        while True:
            try:
                logfile = open(
                    os.path.join(
                        logdir,
                        datetime.now().strftime("%Y-%m-%d-%H:%M")
                        + f"-{os.getpid()}{extra}.log",
                    ),
                    "x",
                )
                break
            except FileExistsError:
                counter += 1
                assert counter < 10
                extra = f"-{counter}"

        # reopen with line buffering
        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)
        sys.stderr = os.fdopen(sys.stderr.fileno(), "w", 1)

        # redirect stdout and stderr to the log file opened above
        os.dup2(logfile.fileno(), sys.stdout.fileno())
        os.dup2(logfile.fileno(), sys.stderr.fileno())

    targets = [
        (f, ex, n)
        for ex in EXPERIMENTS
        for n in corpus_for_experiment(ex)
        for f in [reduce_with_hypothesis, reduce_badly_with_hypothesis, reduce_with_picire, reduce_with_creduce, final_normalized_hypothesis_then_lines]
    ]

    random.shuffle(targets)

    working = tempfile.mkdtemp("working-build-all")
    # black (and possibly other executables?) get a bit upset when trying to
    # run too many copies in parallel and end up corrupting the cache dir. As
    # as result we run each process in its own working directory with its
    # own cache directory.
    cachedir = os.path.join(working, ".cache")
    os.makedirs(cachedir)
    prev_cache = os.environ.get("XDG_CACHE_HOME")
    prev_dir = os.getcwd()
    try:
        os.chdir(working)
        os.environ["XDG_CACHE_HOME"] = cachedir
        backoff = 0.0
        any_skipped = True
        while any_skipped:
            time.sleep(backoff)
            any_skipped = False
            for fn, *args in targets:
                try:
                    fn(*args, skip_locked=True)
                except Locked:
                    any_skipped = True
                    continue

            backoff = (backoff + random.random()) * 2
    finally:
        shutil.rmtree(working)
        os.chdir(prev_dir)
        if prev_cache is not None:
            os.environ["XDG_CACHE_HOME"] = prev_cache
        else:
            del os.environ["XDG_CACHE_HOME"]


@cached_transformation(UTF8)
def initial_normalized(experiment, name):
    normalize = EXPERIMENTS[experiment].normalize_test_case
    generated = generate(experiment, name)
    return normalize(generated)


@cached_transformation(UTF8)
def final_normalized(experiment, name, reducer):
    normalize = EXPERIMENTS[experiment].normalize_test_case
    info = reducer(experiment, name)
    final = info["final"]
    if "hypothesis" in reducer.__name__:
        final = final["generated"]
    return normalize(final)


@cached_transformation(JSON)
def hypothesis_then_delete_lines(experiment, name):
    initial = final_normalized(experiment, name, reduce_with_hypothesis)
    base_predicate = EXPERIMENTS[experiment].calculate_error_predicate(info(experiment, name), check_validity=True)
    generated = generate(experiment, name)
    return run_reducer_base(initial, base_predicate, Picire())


@cached_transformation(UTF8)
def final_normalized_hypothesis_then_lines(experiment, name):
    normalize = EXPERIMENTS[experiment].normalize_test_case
    return normalize(hypothesis_then_delete_lines(experiment, name)["final"])


@cached_transformation(JSON)
def calculate_stats(experiment):
    corpus = corpus_for_experiment(experiment)

    results = []

    for c in corpus:
        info = {"name": c, "experiment": experiment, "initial_size": len(initial_normalized(experiment, c)), "reducers": {}}
        results.append(info)

        for key, fn in [
            ("hypothesis", reduce_with_hypothesis),
            ("hypothesisflat", reduce_badly_with_hypothesis),
            ("creduce", reduce_with_creduce),
            ("picire", reduce_with_picire),
        ]:
            reduction_info = fn(experiment, c)
            reduction_stats = reduction_info["reductionstats"]
            if "hypothesis" in key:
                final = reduction_info["final"]["generated"]
            else:
                final = reduction_info["final"]

            record = {"final_size": len(final_normalized(experiment, c, fn)), "runtime": reduction_stats["end_time"] - reduction_stats["start_time"], "count": sum(v["count"] for v in reduction_stats["statistics"].values()),
                "statistics":  {
                    k: {c: v[c] for c in ("count", "runtime")}
                    for k, v in reduction_stats["statistics"].items()
                    if v["count"] > 0
                }
            }
            if "hypothesis" in key:
                gen_stats = {
                    k: {c: v[c] for c in ("count", "runtime")}
                    for k, v in reduction_info["generationstats"].items()
                    if v["count"] > 0
                }
                gen_stats["total"] = {
                    c: sum(v[c] for v in gen_stats.values()) for c in ("count", "runtime")
                }
                record["generation_statistics"] = gen_stats
            info["reducers"][key] = record
    return results


def sort_key_for_baseline(s):
    # This is fairly arbitrary. The important feature is that a smaller
    # value will always sort first.
    return (len(s), s.count("\n"), len(set(s)), s)


@cached_transformation(UTF8)
def calculate_baseline_from(experiment, reducer, start):
    experiment = EXPERIMENTS[experiment]
    generated = generate(experiment, start)

    best = experiment.normalize_test_case(generated)

    def predicate(source):
        nonlocal best
        result = experiment.check_validity(source)
        if result == Classification.VALID:
            formatted = experiment.normalize_test_case(source)
            if sort_key_for_baseline(formatted) < sort_key_for_baseline(best):
                best = formatted
            return Classification.INTERESTING
        return result

    assert predicate(best) == Classification.INTERESTING

    if reducer == "hypothesis":
        def test_function(data):
            if predicate(data.draw(experiment.generator)) == Classification.INTERESTING:
                data.mark_interesting()

        buffer = raw_buffer(experiment, start)

        runner = eng.ConjectureRunner(
            test_function,
            settings=settings(
                database=None,
                max_examples=1,
                suppress_health_check=HealthCheck.all(),
                deadline=None,
                verbosity=Verbosity.debug,
                buffer_size=BUFFER_SIZE,
            ),
            random=Random(int.from_bytes(hashlib.sha1(buffer).digest(), "big")),
        )
        runner.cached_test_function(buffer)
        runner.shrink_interesting_examples()
    else:
        assert reducer in ('creduce', 'picire')
        if reducer == 'creduce':
            r = CReduce(not_c=experiment.name != 'csmith')
        else:
            r = Picire()
        r.run(best, predicate)

    return best


@cached_transformation(UTF8)
def baseline(experiment, reducer):
    return min([calculate_baseline_from(experiment, reducer, source) for source in corpus_for_experiment(experiment)], key=sort_key_for_baseline)


@main.command()
@click.option("--jobs", default=-1)
@click.option("--seed", default=None, type=int)
def build(jobs, seed):
    import hypothesis
    if jobs <= 0:
        jobs = cpu_count()
    if seed is not None:
        random.seed(seed)

    if jobs > 1:
        fork_pool(n_workers=jobs, run=build_all)
    else:
        build_all()

@main.command()
def show_stats():
    data = os.path.join(ROOT, "data")
    mkdirp(data)
    with open(os.path.join(data, "reduction-stats.jsons"), "w") as o:
        for e in EXPERIMENTS:
            for s in calculate_stats(e):
                t = json.dumps(s)
                print(t)
                print(t, file=o)

@main.command()
@click.argument('experiment')
@click.argument('source')
@click.option('--buffer-size', default=settings().buffer_size)
@click.option('--count', default=200)
def import_corpus(experiment, source, buffer_size, count):
    random.seed(0)
    targets = os.listdir(source)
    targets.sort()
    random.shuffle(targets)

    try:
        os.unlink(os.path.join(CORPORA, experiment + '.tar'))
    except FileNotFoundError:
        pass

    shutil.rmtree(os.path.join(CORPORA, experiment), ignore_errors=True)
    mkdirp(os.path.join(CORPORA, experiment))

    experiment = EXPERIMENTS[experiment]

    completed = 0
    for f in targets:
        f = os.path.join(source, f)
        if os.stat(f).st_size > buffer_size:
            continue
        print(f)
        try:
            with open(f, 'rb') as i:
                buf = i.read()
            gen = ConjectureData.for_buffer(buf).draw(experiment.generator)
            info = experiment.calculate_info(gen)
            error_pred = experiment.calculate_error_predicate(info)
        except Exception:
            traceback.print_exc()
            continue
        with open(os.path.join(CORPORA, experiment.name, os.path.basename(f)), "wb") as o:
            o.write(buf)
        completed += 1
        if completed >= count:
            break

    subprocess.check_call(["apack", experiment.name + ".tar", experiment.name], cwd=CORPORA)
    shutil.rmtree(os.path.join(CORPORA, experiment.name))


@main.command()
@click.argument('experiment')
@click.option('--buffer-size', default=settings().buffer_size)
@click.option('--count', default=200)
@click.option('--seed', default=0)
def generate_corpus(experiment, seed, buffer_size, count):
    random.seed(seed)

    try:
        os.unlink(os.path.join(CORPORA, experiment + '.tar'))
    except FileNotFoundError:
        pass

    shutil.rmtree(os.path.join(CORPORA, experiment), ignore_errors=True)
    mkdirp(os.path.join(CORPORA, experiment))

    experiment = EXPERIMENTS[experiment]

    completed = 0
    while completed < count:
        try:
            data = ConjectureData(
                draw_bytes=lambda data, n: uniform(random, n),
                max_length=buffer_size,
            )
            gen = experiment.generator(data)
            info = experiment.calculate_info(gen)
            error_pred = experiment.calculate_error_predicate(info)
        except StopTest:
            continue
        except Exception:
            continue
        print(info)
        with open(os.path.join(CORPORA, experiment.name, hashlib.sha1(data.buffer).hexdigest()[:16]), "wb") as o:
            o.write(data.buffer)
        completed += 1

    subprocess.check_call(["apack", experiment.name + ".tar", experiment.name], cwd=CORPORA)
    shutil.rmtree(os.path.join(CORPORA, experiment.name))


@main.command()
def show_common_deletion():
    for name in corpus_for_experiment("csmith"):
        h = final_normalized("csmith", name, reduce_with_hypothesis)
        p = final_normalized("csmith", name, reduce_with_picire)
        hp = final_normalized_hypothesis_then_lines("csmith", name)
        if len(hp) > len(p):
            print(f"Picire without ({len(p)} bytes):")
            print(p)
            print()
            print(f"Picire with ({len(hp)} bytes):")
            print(hp)
            print("----------------------")
            print()


@main.command()
@click.argument("experiment")
@click.argument("name")
@click.option("--trace-memory/--no-trace-memory", default=False)
@click.option("--target", default="hypothesis", type=click.Choice(("hypothesis", "picire", "creduce")))
def debug(experiment, name, trace_memory, target):
    """Debugging command, primarily intended to help understand pathological
    performance of Hypothesis on a given example."""
    global trace_memory_usage
    if trace_memory:
        trace_memory_usage = True
        tracemalloc.start()

    fn = globals()["reduce_with_" + target]

    result = fn(experiment, name, ignore_cache=True)
    if trace_memory:
        display_top(tracemalloc.take_snapshot())

    print(json.dumps(result))


@main.command()
def check():
    pass


if __name__ == "__main__":
    main()
