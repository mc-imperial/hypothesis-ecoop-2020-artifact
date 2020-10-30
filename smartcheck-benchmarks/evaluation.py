import subprocess
import sys
import os
import re
import attr
import json
from glob import glob
from enum import Enum
import hashlib
from random import Random
from collections import Counter
from tqdm import trange, tqdm
import numpy as np
import numpy.random as npr
import click
from scipy.stats import entropy
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = os.path.dirname(os.path.abspath(__file__))

HELPERS = os.path.join(ROOT, "src")

sys.path.append(HELPERS)

from evalcommon import HS_COMMON_MODULE, ghc, makedirs  # noqa

HASKELL_TEST_FILES = glob(os.path.join(ROOT, 'evaluations', '*', 'Test.hs'))
PYTHON_TEST_FILES = glob(os.path.join(ROOT, 'evaluations', '*', 'test.py'))


def rm_f(f):
    if os.path.exists(f):
        os.unlink(f)


@attr.s()
class Size(object):
    size_haskell = attr.ib()
    size_python = attr.ib()


def once(fn):
    def run():
        if run.has_run:
            return
        fn()
        run.has_run = True
    run.has_run = False
    run.__name__ = fn.__name__
    return run


# Executables are built outside of the mounted directory to work around a
# docker on Windows bug where the executable sometimes disappears if you run
# it a lot. Really.
TEST_EXECUTABLES = os.path.expanduser('/tmp/.shrink-evaluations')


def name_for_file(fp):
    dirname = os.path.dirname(fp)
    assert os.path.basename(os.path.dirname(dirname)) == 'evaluations'
    return os.path.basename(dirname)


def eval_dir(name):
    return os.path.join(ROOT, 'evaluations', name)


COMPILED = set()


def haskell_test_executable(name, target):
    if target == Target.SmartCheck:
        flags = ['-DUSE_SMARTCHECK']
        basename = 'TestSmart'
    elif target == Target.QuickCheckCustom:
        flags = ['-DUSE_CUSTOM_SHRINKER']
        basename = 'TestQCCustom'
    else:
        assert target == Target.QuickCheck
        flags = []
        basename = 'TestQC'

    result = os.path.join(TEST_EXECUTABLES, name, basename)

    key = (name, target)
    if key in COMPILED:
        return result

    dirname = os.path.dirname(result)
    makedirs(dirname)

    source = os.path.join(ROOT, 'evaluations', name, 'Test.hs')

    files = [HS_COMMON_MODULE]
    support = os.path.join(ROOT, 'evaluations', name, 'Support.hs')
    if os.path.exists(support):
        files.append(support)
    files.append(source)
    ghc(*files, '-o', result, *flags)
    COMPILED.add(key)
    assert os.path.exists(result)
    return result


HYPOTHESIS_OUTPUT = re.compile(
    r'Falsifying example: test[a-z_]*\(\s*\w+=(.+)\)$'
)

HYPOTHESIS_EVALUATION_OUTPUT = re.compile(
    r'EVALUATIONS: ([0-9]+)$'
)


def run_python_example(name, seed, shrink):
    eval_file = os.path.join(eval_dir(name), 'test.py')
    env = dict(os.environ)
    if not shrink:
        env["SHRINK"] = "false"
    env['SEED'] = str(seed)
    env['PYTHONPATH'] = HELPERS
    output = subprocess.check_output(
        [sys.executable, eval_file], env=env,
        encoding='utf-8',
    )
    results = []
    seen_start = False
    evaluations = None
    for l in output.splitlines():
        m = HYPOTHESIS_EVALUATION_OUTPUT.match(l)
        if m is not None:
            evaluations = int(m[1])
            break
        if 'Falsifying example: ' in l:
            seen_start = True
        if seen_start:
            results.append(l.strip())
    assert evaluations is not None, output
    contents = ' '.join(results).strip()
    m = HYPOTHESIS_OUTPUT.match(contents)
    contents = m[1].strip()
    assert contents[-1] == ","
    contents = contents[:-1]
    assert m is not None, output
    return {'contents': contents, 'evaluations': evaluations, "seed": seed}


COMPILED = set()

SIZES = {}


def size_functions(name):
    try:
        return SIZES[name]
    except KeyError:
        pass

    evalfile = os.path.join(eval_dir(name), 'evaluation.py')
    with open(evalfile) as i:
        size_code = i.read()

    namespace = {}
    namespace['__file__'] = evalfile
    exec(size_code, namespace)
    try:
        size = namespace['size']
        result = Size(size_python=size, size_haskell=size)
    except KeyError:
        result = Size(
            size_python=namespace['size_python'],
            size_haskell=namespace['size_haskell']
        )
    SIZES[name] = result
    return result


def python_eval_file(name):
    return os.path.join(eval_dir(name), 'test.py')


class Target(Enum):
    Hypothesis = 0
    QuickCheck = 3
    QuickCheckCustom = 5
    SmartCheck = 4


def run_qc_example_with_custom(name, seed, shrink):
    return run_qc_example(name, seed, shrink, target=Target.QuickCheckCustom)


def run_qc_example(name, seed, shrink, target=Target.QuickCheck):
    eval_executable = haskell_test_executable(
        name=name, target=target,
    )
    env = dict(os.environ)
    env['SEED'] = str(seed)
    if not shrink:
        env["SHRINK"] = "false"
    output = subprocess.check_output(
        [eval_executable], env=env,
        encoding='utf-8', stderr=subprocess.DEVNULL,
    )

    seen_failures = False

    evaluations = 0
    falsifying_line = -1
    for i, l in enumerate(output.splitlines()):
        if l.strip() == 'Failed:':
            seen_failures = True
        if l.strip() in ('Failed:', 'Passed:') and seen_failures:
            evaluations += 1
        if 'Failed!' in l:
            falsifying_line = i
            break
    if falsifying_line < 0:
        raise ValueError("No 'Failed!' line in %r" % (output,))
    return {
        'contents': '\n'.join(output.splitlines()[falsifying_line+1:]).strip(),
        'evaluations': evaluations,
        'seed': seed,
    }


def run_smartcheck_example(name, seed, shrink):
    eval_executable = haskell_test_executable(
        name=name, target=Target.SmartCheck
    )
    assert os.path.exists(eval_executable)
    env = dict(os.environ)
    env['SEED'] = str(seed)
    if not shrink:
        env["SHRINK"] = "false"
    result = subprocess.run(
        [eval_executable], env=env, encoding='utf-8', input='q',
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    result.check_returncode()
    gathered = []
    gathering = False
    for l in result.stdout.splitlines():
        if l.strip() == 'Attempt to find a new counterexample?':
            break
        if gathering:
            gathered.append(l)
        elif l.strip() == '*** Smart-shrunk value:':
            gathering = True
    else:
        raise ValueError(
            "Expected to find counter-example in %r" % (result.stdoutoutput,)
        )
    return {'contents': '\n'.join(gathered).strip(), 'seed': seed}


TO_EVAL = {
    Target.Hypothesis: run_python_example,
    Target.QuickCheckCustom: run_qc_example_with_custom,
    Target.QuickCheck: run_qc_example,
    Target.SmartCheck: run_smartcheck_example,
}


@attr.s()
class Stat(object):
    estimate = attr.ib()
    lower_confidence = attr.ib()
    upper_confidence = attr.ib()


def empirical_normalization(labels):
    stats = np.unique(labels, return_counts=True)[1]
    n = len(labels)
    probs = stats / n
    return entropy(probs)


N_RUNS = 1000
N_BOOTSTRAPS = 10000


def run_evaluation(name, target):
    evalf = TO_EVAL[target]

    rnd = Random(int.from_bytes(
        hashlib.sha1((target.name + ":" + name).encode('ascii')).digest(),
        byteorder='big'
    ))

    results = []

    def run_seed(seed):
        original = evalf(name, seed, shrink=False)["contents"]
        experiment = evalf(name, seed, shrink=True)
        experiment["original"] = original
        return experiment
        
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as tpe:
        with tqdm(total=N_RUNS) as pb:
            results = [
                run_seed(rnd.getrandbits(32))
            ]
            pb.update(1)

            futures = [
                tpe.submit(run_seed, rnd.getrandbits(32)) for _ in range(N_RUNS - 1)
            ]

            for f in as_completed(futures):
                results.append(f.result())
                pb.update(1)

            assert len(results) == N_RUNS
    return results


PYTHON_TARGETS = (
    Target.Hypothesis,
)


def size_function_for(name, target):
    if isinstance(target, str):
        target = Target[target]
    if target in PYTHON_TARGETS:
        return size_functions(name).size_python
    else:
        return size_functions(name).size_haskell


def sizes_from_counts(name, target, counts=None):
    if counts is None:
        counts = counts_for(name, target)

    sizes = np.zeros(shape=N_RUNS, dtype=int)


    if target in PYTHON_TARGETS:
        sizef = size_functions(name).size_python
    else:
        sizef = size_functions(name).size_haskell

    i = 0
    for r, n in counts.items():
        sizes[i:i+n] = sizef(r)
        i += n
    return sizes


def stat_from_bootstraps(empirical, bootstraps):
    # This is a pivot bootstrap method and I'd be lying if I said I fully
    # understood it. The idea is as follows though: We have some function F
    # of the true distribution. We expect the empirical distribution to be very
    # close to the true distribution. BUT just taking the value of F on the
    # empirical distribution may be a biased estimator. So we use the bootstrap
    # to figure out how biased it is, and use that to correct. A similar effect
    # applies to our confidence intervals.
    base = 2 * empirical
    return Stat(
        estimate=base - np.mean(bootstraps),
        lower_confidence=base - np.percentile(bootstraps, 97.5),
        upper_confidence=base - np.percentile(bootstraps, 2.5),
    )


def mean_size(name, target):
    counts = counts_for(name, target)
    sizes = sizes_from_counts(name, target, counts)
    estimate = np.mean(sizes)
    bootstrap_results = np.random.choice(
        sizes, size=(len(sizes), N_BOOTSTRAPS)
    ).mean(0)

    return stat_from_bootstraps(estimate, bootstrap_results)


def unpredictability(name, target):
    counts = counts_for(name, target)
    labels = np.zeros(shape=N_RUNS, dtype=int)
    unique_id = {}
    i = 0
    for r, n in counts.items():
        labels[i:i+n] = unique_id.setdefault(r, len(unique_id))
        i += n
    assert i == N_RUNS

    estimate = empirical_normalization(labels)

    bootstrap_results = np.array([
        empirical_normalization(npr.choice(labels, size=len(labels)))
        for _ in range(N_BOOTSTRAPS)
    ])

    return stat_from_bootstraps(estimate, bootstrap_results)


def size_quantile95(name, target):
    counts = counts_for(name, target)

    sizes = sizes_from_counts(name, target, counts)

    percent = 95

    estimate = np.percentile(sizes, percent)

    # NB: There are two different qs here that happen to have the same value.
    # We're estimating the 95% of the sizes, but we're also determining a 95%
    # confidence interval of that estimate.
    bootstrap_results = np.percentile(np.random.choice(
        sizes, size=(len(sizes), N_BOOTSTRAPS)
    ), q=percent, axis=0)

    assert len(bootstrap_results) == N_BOOTSTRAPS

    return stat_from_bootstraps(estimate, bootstrap_results)


def sut_evaluations(name, target):
    evalfile = eval_file(name, target)
    with open(evalfile) as i:
        values = json.loads(i.read())
    evaluations = np.zeros(shape=len(values), dtype=int)
    assert len(values) == len(evaluations)
    for i, v in enumerate(values):
        if 'evaluations' not in v:
            return None
        evaluations[i] = int(v['evaluations'])
    return evaluations



def mean_sut_evaluations(name, target):
    evaluations = sut_evaluations(name, target)
    if evaluations is None:
        return None
    estimate = np.mean(evaluations)
    bootstrap_results = np.random.choice(
        evaluations, size=(len(evaluations), N_BOOTSTRAPS)
    ).mean(0)

    return stat_from_bootstraps(estimate, bootstrap_results)


def q95_sut_evaluations(name, target):
    evaluations = sut_evaluations(name, target)
    if evaluations is None:
        return None
    percent = 95

    estimate = np.percentile(evaluations, percent)

    # NB: There are two different qs here that happen to have the same value.
    # We're estimating the 95% of the evaluations, but we're also determining a 95%
    # confidence interval of that estimate.
    bootstrap_results = np.percentile(np.random.choice(
        evaluations, size=(len(evaluations), N_BOOTSTRAPS)
    ), q=percent, axis=0)

    assert len(bootstrap_results) == N_BOOTSTRAPS

    return stat_from_bootstraps(estimate, bootstrap_results)


EVALUATIONS = {}

for f in HASKELL_TEST_FILES:
    name = os.path.basename(os.path.dirname(f))
    ev = EVALUATIONS.setdefault(name, [])
    ev.append(Target.QuickCheck)
    ev.append(Target.SmartCheck)
    with open(f, 'r') as i:
        if 'USE_CUSTOM_SHRINKER' in i.read():
            ev.append(Target.QuickCheckCustom)

for f in PYTHON_TEST_FILES:
    name = os.path.basename(os.path.dirname(f))
    ev = EVALUATIONS.setdefault(name, [])
    for e in PYTHON_TARGETS:
        ev.append(e)


@click.group()
def main():
    pass


@main.command()
@click.argument('names', nargs=-1)
@click.option('--target', multiple=True)
@click.option('--rebuild/--no-rebuild', default=False)
@click.option('--runs', default=0)
def build(names, target, rebuild, runs):
    global N_RUNS
    if runs > 0:
        N_RUNS = runs
    if target:
        targets = [Target[name] for name in target]
    else:
        targets = list(Target)

    if not names:
        names = EVALUATIONS

    selected = sorted(names)

    for name in selected:
        evaldata = os.path.join(
            os.path.dirname(__file__), "data", "evaluations", name
        )
        makedirs(evaldata)
        for t in EVALUATIONS[name]:
            if t not in targets:
                continue
            evalfile = os.path.join(evaldata, t.name + '.json')
            print(evalfile)
            if os.path.exists(evalfile) and not rebuild:
                with open(evalfile) as i:
                    data = json.loads(i.read())
                count = len(data)
                if count >= N_RUNS:
                    continue
            print(name, t)
            counts = run_evaluation(name, t)
            with open(evalfile, 'w') as o:
                o.write(json.dumps(counts))


def eval_file(name, target):
    return os.path.join(
        ROOT, "data", "evaluations", name, target.name + '.json'
    )


COUNTS = {}


def counts_for(name, target):
    key = (name, target)

    try:
        return COUNTS[key]
    except KeyError:
        pass

    evalfile = eval_file(name, target)
    with open(evalfile) as i:
        counts = Counter()
        for m in json.loads(i.read()):
            counts[m['contents']] += 1
        return COUNTS.setdefault(key, counts)


NAMES = {
    Target.Hypothesis: 'Hypothesis',
}


def name_of(target):
    return NAMES.get(target, target.name)


@main.command()
@click.argument('names', nargs=-1)
@click.option('--only')
def analyze(names, only):
    if only is not None:
        only = set(s.strip().lower() for s in only.split(','))

    for fn in [
        mean_size,
        size_quantile95,
        unpredictability,
        mean_sut_evaluations,
        q95_sut_evaluations,
    ]:
        s = fn.__name__
        if only and s.lower() not in only:
            continue

        for name in sorted(names or EVALUATIONS):
            for target in Target:
                evalfile = os.path.join(
                    ROOT, "data", "evaluations", name, target.name + '.json'
                )
                if not os.path.exists(evalfile):
                    continue
                npr.seed(0)
                result = fn(name, target)
                if result is None:
                    continue
                print(json.dumps({
                    'name': name, 'target': NAMES.get(target, target.name),
                    'stat': s,
                    'value': attr.asdict(result),
                }))


@main.command()
def dump_data_points():
    evaldir = os.path.join(os.path.dirname(__file__), "data", "evaluations")

    with open(os.path.join(os.path.dirname(__file__), "data", "evaluation-data.jsons"), "w") as o:
        for benchmark in os.listdir(evaldir):
            bm_dir = os.path.join(evaldir, benchmark)
            for target in os.listdir(bm_dir):
                if not target.endswith('.json'):
                    continue
                target_file = os.path.join(bm_dir, target)
                with open(target_file) as i:
                    data = json.loads(i.read())
                target = target.replace('.json', '')
                try:
                    sizef = size_function_for(benchmark, target)
                except KeyError:
                    continue
                for observation in data:
                    print(json.dumps({
                        'benchmark': benchmark,
                        'target': target,
                        'size': sizef(observation["contents"]),
                        'original_size': sizef(observation["original"]),
                        'evaluations': observation.get("evaluations"),
                        'seed': observation["seed"],
                    }), file=o)
    

def p_value_mean(name, t1, t2):
    assert t1 != t2

    sizes1 = sizes_from_counts(name, t1)
    sizes2 = sizes_from_counts(name, t2)

    n = len(sizes1)
    assert n == len(sizes2)

    baseline = abs(np.mean(sizes1) - np.mean(sizes2))

    joint = np.concatenate([sizes1, sizes2])

    npr.seed(0)
    tot = 0
    for _ in trange(N_BOOTSTRAPS):
        npr.shuffle(joint)
        r = abs(np.mean(joint[:n]) - np.mean(joint[n:]))
        if r >= baseline:
            tot += 1
    # See "Permutation P-values Should Never Be Zero",
    # Belinda Phipson and Gordon K. Smyth
    return (1 + float(tot)) / (1 + N_BOOTSTRAPS)


@main.command()
@click.argument('names', nargs=-1)
def significance(names):
    if not names:
        names = list(EVALUATIONS)
    names = sorted(names)

    for name in names:
        for t1 in EVALUATIONS[name]:
            for t2 in EVALUATIONS[name]:
                if t1.name < t2.name:
                    p = p_value_mean(name, t1, t2)
                    print('\t'.join((name, name_of(t1), name_of(t2), str(p))))


if __name__ == '__main__':
    main()
