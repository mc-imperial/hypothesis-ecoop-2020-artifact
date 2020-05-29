import click
from functools import wraps
import os
import shutil
import hashlib
import subprocess
import sys
from threading import RLock
import binascii
import json
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import attr
from collections import defaultdict
import random
from tqdm import tqdm


os.environ["PYTHONHASHSEED"] = "0"


HERE = os.path.abspath(os.path.dirname(__file__))


WORKING_DIR = os.path.join(HERE, ".working")
EXPERIMENTS_DIR = os.path.join(HERE, "experiments")

WORKER_PROGRAM = os.path.join(HERE, "tstl_hypothesis.py")


EXPERIMENTS = os.listdir(EXPERIMENTS_DIR)


TSTL = os.path.join(os.path.dirname(sys.executable), "tstl")
TSTL_REPLAY = os.path.join(os.path.dirname(sys.executable), "tstl_replay")


N_STEPS = 100


def mkdirp(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def cached(fn):
    cache = {}
    lock = RLock()
    @wraps(fn)
    def accept(*args):
        try:
            return cache[args]
        except KeyError:
            pass
        try:
            lock.acquire()
            try:
                return cache[args]
            except KeyError:
                return cache.setdefault(args, fn(*args))
        finally:
            lock.release()
    return accept



@cached
def virtualenv(experiment):
    exdir = os.path.join(EXPERIMENTS_DIR, experiment)
    requirements_file =  os.path.join(exdir, "requirements.txt")

    main_requirements_file = os.path.join(HERE, "requirements.txt")
    with open(main_requirements_file, 'rb') as i:
        main_reqs = i.read()

    with open(requirements_file, 'rb') as i:
        name = f"{experiment}-{hashlib.sha1(main_reqs + i.read()).hexdigest()[:10]}"

    virtualenvs = os.path.join(WORKING_DIR, "virtualenvs")
    mkdirp(virtualenvs)

    dest = os.path.join(virtualenvs, name)

    dest_python = os.path.join(dest, "bin", "python")

    if not os.path.exists(dest_python):
        try:
            subprocess.check_call([
                sys.executable, "-m", "virtualenv", dest
            ])
            assert os.path.exists(dest_python)
            subprocess.check_call([
                dest_python, "-m", "pip", "install", "-r", main_requirements_file
            ])
            subprocess.check_call([
                dest_python, "-m", "pip", "install", "-r", requirements_file
            ])
        except BaseException as e:
            shutil.rmtree(dest, ignore_errors=True)
            raise e

    return dest


def run_tstl_program(experiment, program, seed=0, hash_seed=0):
    python = os.path.join(virtualenv(experiment), "bin", "python")
    exdir = os.path.join(EXPERIMENTS_DIR, experiment)

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hash_seed)
    env["PYTHONPATH"] = exdir

    sp = subprocess.Popen(
        [python, os.path.join(HERE, "tstl_controller.py")],
        env=env, universal_newlines=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=sys.stderr,
    )
    try:
        out, _ = sp.communicate(json.dumps([
            program, seed
        ]), timeout=5 + 2 * len(program))
        assert sp.returncode == 0
        return json.loads(out)
    finally:
        if sp.returncode is None:
            sp.kill()



@attr.s(slots=True)
class FlakyTree(object):
    count = attr.ib(default=0)
    ever_passed = attr.ib(default=False)
    failures = attr.ib(default=attr.Factory(set))
    disabled = attr.ib(default=attr.Factory(set))
    children = attr.ib(default=attr.Factory(lambda: defaultdict(FlakyTree)), repr=False)
    timed_out = attr.ib(default=False)

    @property
    def flaky(self):
        if self.failures and self.ever_passed:
            return True
        if len(self.failures) > 1:
            return True
        if self.timed_out:
            return True
        return False



class FlakinessDetector(object):
    def __init__(self, experiment):
        self.experiment = experiment

        self.__lock = RLock()
        self.__root = FlakyTree()
        self.__intern = {}
        self.__random = random.Random()

        replay_dir = os.path.join(WORKING_DIR, "flakiness")
        mkdirp(replay_dir)

        replay_file = os.path.join(replay_dir, experiment)
        if os.path.exists(replay_file):
            with open(replay_file) as i:
                for line in i:
                    self.__incorporate_no_log(json.loads(line))

        self.__replay_logger = open(replay_file, "a")

    def is_flaky(self, program):
        try:
            self.__lock.acquire()
            node = self.__root
            assert not node.flaky, "Root is flaky!"

            trail = []

            for p in program:
                if p not in node.disabled:
                    node = node.children[p]
                    trail.append(node) 
                    if node.flaky:
                        return True
        finally:
            self.__lock.release()

        while node.count < 3:
            result = run_tstl_program(self.experiment, program, seed=self.__random.getrandbits(64), hash_seed=self.__random.getrandbits(16))
            if result["outcome"] is not None:
                # Indicates a bad TSTL program
                assert result["outcome"][0] != "KeyError" or "tstl_controller" not in result["outcome"][1]
            self.__incorporate(result)
            if any(n.flaky for n in trail):
                return True
        return False

    def __incorporate(self, result):
        try:
            self.__lock.acquire()
            self.__replay_logger.write(json.dumps(result))
            self.__replay_logger.write("\n")
            self.__replay_logger.flush()
            self.__incorporate_no_log(result)
        finally:
            self.__lock.release()
            
    def __incorporate_no_log(self, result):
        node = self.__root
        node.count += 1

        read = result["read"]

        for i in range(read):
            if node.flaky:
                break

            if i + 1 < read:
                node.ever_passed = True

            step = result["source_program"][i]
            step = self.__intern.setdefault(step, step)
            if result["enabled"][i]:
                # We successfully attempted to execute a step after it so it
                # must not have failed.
                node = node.children[step]
                node.count += 1
            else:
                node.disabled.add(step)
        if result["outcome"] is None:
            node.ever_passed = True
        else:
            outcome = tuple(result["outcome"])
            outcome = self.__intern.setdefault(outcome, outcome)
            node.failures.add(outcome)

@cached
def tstl_harness(experiment):
    exdir = os.path.join(EXPERIMENTS_DIR, experiment)
    tstl_file =  os.path.join(exdir, "test.tstl")
    dest = os.path.join(exdir, "sut.py")

    if not os.path.exists(dest):
        subprocess.check_call([
            TSTL, tstl_file, "--output", dest
        ])

    return dest


def generate(experiment, seed):
    log_dir = os.path.join(WORKING_DIR, "generated", experiment)
    mkdirp(log_dir)
    log_file = os.path.join(log_dir, f"{seed}-{N_STEPS}.jsons")
    if not os.path.exists(log_file):
        tmp_log = log_file + "." + binascii.hexlify(os.urandom(8)).decode('ascii')
        python = os.path.join(virtualenv(experiment), "bin", "python")
        sut_path = tstl_harness(experiment)
        subprocess.check_call([
            python, "-u", WORKER_PROGRAM,
            "--sut-file",  sut_path,
            "--log-file", tmp_log,
            "--seed", str(seed),
            "--n-steps", str(N_STEPS),
            "generate",
        ])
        try:
            os.rename(tmp_log, log_file)
        except FileExistsError:
            os.unlink(tmp_log)
    with open(log_file) as i:
        for l in i:
            pass
        return json.loads(l)


def shrink(experiment, seed, delete_only=False):
    generated = generate(experiment, seed)

    if is_flaky(experiment, generated[-1]):
        return None

    assert generated[3] == "INTERESTING"

    log_dir = os.path.join(WORKING_DIR, "shrunk", experiment)
    mkdirp(log_dir)
    log_file = os.path.join(log_dir, f"{seed}-{N_STEPS}{'-do' if delete_only else ''}.jsons")
    if not os.path.exists(log_file):
        tmp_log = log_file + "." + binascii.hexlify(os.urandom(8)).decode('ascii')
        python = os.path.join(virtualenv(experiment), "bin", "python")
        sut_path = tstl_harness(experiment)
        subprocess.check_call([
            python, "-u", WORKER_PROGRAM,
            "--sut-file",  sut_path,
            "--log-file", tmp_log,
            "--seed", str(seed),
            "--n-steps", str(N_STEPS),
            "shrink",
            "--delete-only" if delete_only else  "--full",
            generated[-2],
        ])
        try:
            os.rename(tmp_log, log_file)
        except FileExistsError:
            os.unlink(tmp_log)

    target = generated[4]

    with open(log_file) as i:
        calls = 0
        for l in i:
            if l.startswith("//"):
                continue
            calls += 1
            data = json.loads(l.strip())
            if data[3] == "INTERESTING" and data[4] == target:
               result = data[-1]

        return {
            "result": result, "calls": calls
        }


def external_shrink(experiment, seed):
    source = generate(experiment, seed)

    if is_flaky(experiment, source[-1]):
        return None

    log_dir = os.path.join(WORKING_DIR, "external-shrunk", experiment)
    mkdirp(log_dir)
    log_file = os.path.join(log_dir, f"{seed}-{N_STEPS}.jsons")
    if not os.path.exists(log_file):
        exdir = os.path.join(EXPERIMENTS_DIR, experiment)
        tmp_log = log_file + "." + binascii.hexlify(os.urandom(8)).decode('ascii')
        python = os.path.join(virtualenv(experiment), "bin", "python")
        sut_path = tstl_harness(experiment)

        with tempfile.TemporaryDirectory() as d:
            input_program = os.path.join(d, "program.test")
            with open(input_program, 'w') as o:
                o.write('\n'.join(source[-1]))
            subprocess.check_call([
                python, "-u", os.path.join(HERE, "tstl_reducer.py"),
                "--sut-file",  tstl_harness(experiment),
                "--log-file", tmp_log,
                "--program-file", input_program,
            ])
        try:
            os.rename(tmp_log, log_file)
        except FileExistsError:
            os.unlink(tmp_log)
    with open(log_file) as i:
        return json.loads(i.read())


def validate_shrink_for_flakiness(experiment, seed):
    result = shrink(experiment, seed)
    if result is None:
        raise ValidationError("Generated example was flaky")

    shrinks = -1

    intended_reason = None

    programs = []

    for line in result.splitlines():
        line = line.strip()
        if line.startswith("//"):
            continue

        data = json.loads(line)
        if data[3] != "INTERESTING":
            continue

        reason = data[4]
        if intended_reason is None:
            intended_reason = reason
        elif reason != intended_reason:
            continue

        program = tuple(data[-1])
        if not programs or program != programs[-1]:
            programs.append(program)

    if is_flaky(experiment, programs[-1]):
        raise ValidationError(f"Final result was flaky")

    if len(programs) > 2:
        indices = range(1, len(programs) - 1)
        if len(indices) > 5:
            indices = random.Random(seed).sample(indices, 5)

        for i in sorted(indices):
            if is_flaky(experiment, programs[i]):
                raise ValidationError(f"Shrink {i} was flaky")


class ValidationError(Exception):
    pass


def validate_tstl(experiment, source):
    error_class, error_file, line_no = source[4]

    tstl_harness(experiment)
    with tempfile.TemporaryDirectory() as d:
        test_file = os.path.join(d, "validate.test")
        with open(test_file, "w") as o:
            o.write('\n'.join(source[-1]) + '\n')

        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(EXPERIMENTS_DIR, experiment)

        try:
            subprocess.check_output([
                os.path.join(virtualenv(experiment), "bin", "tstl_replay"),
                test_file,
            ], env=env, cwd=d, stderr=subprocess.PIPE, universal_newlines=True)
            raise ValidationError(
                "Expected TSTL program to fail but it did not"
            )
        except subprocess.CalledProcessError as e:
            if error_class not in e.stdout:
                sys.stdout.write(e.stdout)
                raise ValidationError(f"Expected error type {error_class}")


@click.group()
def experiments():
    pass


@experiments.command()
def console():
    import IPython
    IPython.embed()
    

@experiments.command()
@click.argument('snippet')
def snippet(snippet):
    res = eval(snippet, globals())
    if res is not None:
        print(res)


@cached
def standard_seeds(ex):
    return tuple(int.from_bytes(
            hashlib.sha1(f"{ex}::{seed_base}".encode('utf-8')).digest(),
            'big',
        )
        for seed_base in range(3000)
    )


@cached
def flakiness_detector(ex):
    return FlakinessDetector(ex)


def is_flaky(ex, program):
    return flakiness_detector(ex).is_flaky(program)


@experiments.command()
@click.option('--parallelism', default=10)
@click.option('--delete-only/--full', default=False)
def shrink_all(parallelism, delete_only):
    def run_task(ex, seed):
        try:
            if not delete_only:
                external_shrink(ex, seed)
            shrink(ex, seed, delete_only=delete_only)
        except (subprocess.CalledProcessError, ValidationError):
            traceback.print_exc()

    if parallelism <= 1:
        for ex in EXPERIMENTS:
            for seed in standard_seeds(ex):
                run_task(ex, seed)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            for ex in tqdm(EXPERIMENTS):
                seeds = standard_seeds(ex)
                futures = []
                with tqdm(total=len(seeds)) as pb:
                    for seed in seeds:
                        futures.append(executor.submit(run_task, ex, seed))
                    for future in as_completed(futures):
                        pb.update(1)


@experiments.command()
@click.option('--parallelism', default=10)
def dump_data(parallelism):
    def run_task(ex, seed):
        source = generate(ex, seed)

        try:
            if is_flaky(ex, source[-1]):
                print(f"Rejecting {seed} due to flaky generated example", file=sys.stderr)
                return

            assert source[3] == "INTERESTING"
            target = source[4]

            externally_shrunk = external_shrink(ex, seed)
            if is_flaky(ex, externally_shrunk["result"]):
                print(f"Rejecting {seed} due to flaky external shrink", file=sys.stderr)
                return

            internally_shrunk_do = shrink(ex, seed, delete_only=True)
            if is_flaky(ex, internally_shrunk_do["result"]):
                print(f"Rejecting {seed} due to flaky delete only internal shrink", file=sys.stderr)
                return

            internally_shrunk = shrink(ex, seed)
            if is_flaky(ex, internally_shrunk["result"]):
                print(f"Rejecting {seed} due to flaky internal shrink", file=sys.stderr)
                return

        except subprocess.SubprocessError:
            return None

        for s in [externally_shrunk, internally_shrunk, internally_shrunk_do]:
            s["result"] = len(s["result"])

        return {
            "experiment": ex,
            "seed": seed,
            "bug": target,
            "original": len(source[-1]),
            "internal": internally_shrunk,
            "internal_delete_only:": internally_shrunk_do,
            "external": externally_shrunk,
        }


    with ThreadPoolExecutor(max_workers=parallelism) as tp:
        futures = [
            tp.submit(run_task, ex, seed)
            for ex in EXPERIMENTS
            for seed in standard_seeds(ex)
        ]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                print(json.dumps(result))

@experiments.command()
def validate_all_shrinks():
    for ex in tqdm(EXPERIMENTS):
        for seed in tqdm(standard_seeds(ex)):
            try:
                validate_shrink_for_flakiness(ex, seed, 100)
            except ValidationError as e:
                tqdm.write(f"{ex} was flaky for seed {seed}: {e.args[0]}")
            except subprocess.CalledProcessError as e:
                tqdm.write(f"{ex} had error for seed {seed}: {e.args[0]}")


if __name__ == '__main__':
    experiments()
