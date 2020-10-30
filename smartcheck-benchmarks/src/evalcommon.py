import sys
import os
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import traceback


ROOT = os.path.dirname(os.path.dirname(__file__))

try:
    import hypothesis
except ImportError:
    pass
else:
    from hypothesis.internal.conjecture.engine import ConjectureRunner
    from hypothesis.internal.conjecture.data import Status
    from hypothesis import seed, given, settings, HealthCheck, \
        Verbosity, Phase
    from hypothesis.internal.reflection import proxies
    from hypothesis.errors import HypothesisException

    def eval_given(strat):
        def accept(test):
            verbose = os.environ.get('VERBOSE') == 'true'
            shrink = os.environ.get('SHRINK', 'true') == 'true'
            provided_seed = int(os.environ['SEED'])
            if verbose:
                verbosity = Verbosity.debug
            else:
                verbosity = Verbosity.normal

            seen_failure = False
            evaluations = 0

            original_test = test

            @proxies(test)
            def recording_test(*args, **kwargs):
                nonlocal seen_failure, evaluations
                if seen_failure:
                    evaluations += 1
                try:
                    return original_test(*args, **kwargs)
                except HypothesisException:
                    raise
                except Exception:
                    if not seen_failure:
                        seen_failure = True
                        evaluations += 1
                    raise

            test = seed(provided_seed)(
                settings(
                    database=None, max_examples=10**6,
                    suppress_health_check=HealthCheck.all(),
                    verbosity=verbosity, phases=[Phase.generate, Phase.shrink] if shrink else [Phase.generate] 
                )(
                        given(strat)(recording_test)))

            try:
                test()
            except Exception:
                if verbose:
                    traceback.print_exc()
                print("EVALUATIONS:", evaluations)
                print("TEST FAILED")
                return

            print(
                "Expected test to fail with assertion error", file=sys.stderr
            )
            sys.exit(1)
        return accept


def ghc(*args):
    cmd = ['ghc', *args]
    proc = Popen(cmd, stdout=PIPE, stderr=STDOUT, encoding='utf-8')
    output = proc.stdout.read()
    proc.wait()
    if proc.returncode != 0:
        print(output, file=sys.stderr)
        raise CalledProcessError(proc.returncode, cmd)


def makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


SUPPORT_EXECUTABLES = os.path.expanduser('/tmp/cache/support-executables')

HS_COMMON_MODULE = os.path.join(os.path.dirname(__file__), 'EvalCommon.hs')

EXECUTABLES_BUILT = set()


def support_executable(name, executable):
    location = os.path.join(SUPPORT_EXECUTABLES, name)
    target = os.path.join(location, executable)
    key = (name, executable)
    if key in EXECUTABLES_BUILT:
        return target

    source = os.path.join(ROOT, 'evaluations', name, executable + '.hs')

    if os.path.exists(target) and (
        os.path.getctime(target) > os.path.getmtime(source)
    ):
        EXECUTABLES_BUILT.add(key)
        return target

    makedirs(location)
    files = [HS_COMMON_MODULE]
    support = os.path.join(ROOT, 'evaluations', name, 'Support.hs')
    if os.path.exists(support):
        files.append(support)
    files.append(source)
    ghc(*files, '-o', target)
    assert os.path.exists(target)
    EXECUTABLES_BUILT.add(key)
    return target
