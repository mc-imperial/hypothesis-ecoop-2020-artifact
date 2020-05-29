import attr

from hypothesiscsmith import csmith
import tempfile
from reducereval import ROOT
import os
from glob import glob
import subprocess
import hashlib
from reducereval.experiments.experiments import define_experiment
import dateutil.parser
from reducereval.reduction import Classification
import sys


RUNTIME = os.path.join(ROOT, "hypothesis-csmith", "runtime")


TIMEOUT = 5
GCCS = glob("/opt/compiler-explorer/gcc-*/bin/gcc")
CLANGS = glob("/opt/compiler-explorer/clang-*/bin/clang")


def rmf(f):
    try:
        os.unlink(f)
    except FileNotFoundError:
        pass


def run_compiler(compiler_bin, opt, source):
    if not os.path.exists(compiler_bin):
        compiler_bin = NAMES_TO_PATHS[compiler_bin]
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ)
        env["LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"

        prog = os.path.join(d, "in.c")
        with open(prog, "w") as o:
            o.write(source)

        target = os.path.join(d, "a.out")
        args = [compiler_bin, opt, "-I", RUNTIME, prog, "-o", target]
        try:
            try:
                subprocess.run(
                    args,
                    cwd=d,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    timeout=TIMEOUT,
                    env=env,
                    check=True,
                )
                assert os.path.exists(target)
            except subprocess.CalledProcessError as e:
                error = e.stderr
                if (
                    e.returncode < 0
                    or b"internal compiler error" in error
                    or b"Assertion" in error
                ):
                    return {"compiler-crash": e.returncode}
                else:
                    return {"invalid-source": e.returncode}
            except subprocess.TimeoutExpired as e:
                return {"compiler-timeout": ""}
            try:
                return {
                    "success": hashlib.sha1(
                        subprocess.check_output([target, "1"], timeout=TIMEOUT)
                    ).hexdigest()
                }
            except subprocess.TimeoutExpired as e:
                return {"bin-timeout": ""}
            except subprocess.CalledProcessError as e:
                return {"bin-crash": e.returncode}
        finally:
            rmf(target)


OPTIMISATION_LEVELS = ["-O0", "-O1", "-O2", "-Os"]

GCC_RELEASE_TABLE = """
8.3     February 22, 2019
7.4     December 6, 2018
6.5     October 26, 2018
8.2     July 26, 2018
8.1     May 2, 2018
7.3     January 25, 2018
5.5     October 10, 2017
7.2     August 14, 2017
6.4     July 4, 2017
7.1     May 2, 2017
6.3     December 21, 2016
6.2     August 22, 2016
4.9.4   August 3, 2016
5.4     June 3, 2016
6.1     April 27, 2016
5.3     December 4, 2015
5.2     July 16, 2015
4.9.3   June 26, 2015
4.8.5   June 23, 2015
5.1     April 22, 2015
4.8.4   December 19, 2014
4.9.2   October 30, 2014
4.9.1   July 16, 2014
4.7.4   June 12, 2014
4.8.3   May 22, 2014
4.9.0   April 22, 2014
4.8.2   October 16, 2013
4.8.1   May 31, 2013
4.6.4   April 12, 2013
4.7.3   April 11, 2013
4.8.0   March 22, 2013
4.7.2   September 20, 2012
4.5.4   July 2, 2012
4.7.1   June 14, 2012
4.7.0   March 22, 2012
4.4.7   March 13, 2012
4.6.3   March 1, 2012
4.6.2   October 26, 2011
4.6.1   June 27, 2011
4.3.6   June 27, 2011
4.5.3   April 28, 2011
4.4.6   April 16, 2011
4.6.0   March 25, 2011
4.5.2   December 16, 2010
4.4.5   October 1, 2010
4.5.1   July 31, 2010
4.3.5   May 22, 2010
4.4.4   April 29, 2010
4.5.0   April 14, 2010
4.4.3   January 21, 2010
4.4.2   October 15, 2009
4.3.4   August 4, 2009
4.4.1   July 22, 2009
4.4.0   April 21, 2009
4.3.3   January 24, 2009
4.3.2   August 27, 2008
4.3.1   June 6, 2008
4.2.4   May 19, 2008
4.3.0   March 5, 2008
4.2.3   February 1, 2008
4.2.2   October 7, 2007
4.2.1   July 18, 2007
4.2.0   May 13, 2007
4.1.2   February 13, 2007
4.0.4   January 31, 2007
4.1.1   May 24, 2006
4.0.3   March 10, 2006
3.4.6   March 06, 2006
4.1.0   February 28, 2006
3.4.5   November 30, 2005
4.0.2   September 28, 2005
4.0.1   July 7, 2005
3.4.4   May 18, 2005
3.3.6   May 3, 2005
4.0.0   April 20, 2005
3.4.3   November 4, 2004
3.3.5   September 30, 2004
3.4.2   September 6, 2004
3.4.1   July 1, 2004
3.3.4   May 31, 2004
3.4.0   April 18, 2004
3.3.3   February 14, 2004
3.3.2   October 17, 2003
3.3.1   August 8, 2003
3.3     May 13, 2003
3.2.3   April 22, 2003
3.2.2   February 05, 2003
3.2.1   November 19, 2002
3.2     August 14, 2002
3.1.1   July 25, 2002
3.1     May 15, 2002
3.0.4   February 20, 2002
3.0.3   December 20, 2001
3.0.2   October 25, 2001
3.0.1   August 20, 2001
3.0     June 18, 2001
2.95.3  March 16, 2001
2.95.2  October 24, 1999
2.95.1  August 19, 1999
2.95    July 31, 1999
"""

RELEASE_DATES = {}

for l in GCC_RELEASE_TABLE.splitlines():
    if not l:
        continue
    version, date = l.split(maxsplit=1)
    RELEASE_DATES["gcc-" + version] = dateutil.parser.parse(date)

CLANG_RELEASE_TABLE = """
21 Dec 2018 7.0.1
19 Sep 2018 7.0.0
5 Jul 2018  6.0.1
16 May 2018 5.0.2
8 Mar 2018  6.0.0
21 Dec 2017 5.0.1
07 Sep 2017 5.0.0
04 Jul 2017 4.0.1
13 Mar 2017 4.0.0
23 Dec 2016 3.9.1
02 Sep 2016 3.9.0
11 Jul 2016 3.8.1
08 Mar 2016 3.8.0
05 Jan 2016 3.7.1
01 Sep 2015 3.7.0
16 Jul 2015 3.6.2
26 May 2015 3.6.1
27 Feb 2015 3.6.0
02 Apr 2015 3.5.2
20 Jan 2015 3.5.1
03 Sep 2014 3.5.0
19 Jun 2014 3.4.2
07 May 2014 3.4.1
02 Jan 2014 3.4
17 Jun 2013 3.3
20 Dec 2012 3.2
22 May 2012 3.1
01 Dec 2011 3.0
06 Apr 2011 2.9
05 Oct 2010 2.8
27 Apr 2010 2.7
23 Oct 2009 2.6
02 Mar 2009 2.5
09 Nov 2008 2.4
09 Jun 2008 2.3
11 Feb 2008 2.2
26 Sep 2007 2.1
23 May 2007 2.0
19 Nov 2006 1.9
09 Aug 2006 1.8
20 Apr 2006 1.7
08 Nov 2005 1.6
18 May 2005 1.5
09 Dec 2004 1.4
13 Aug 2004 1.3
19 Mar 2004 1.2
17 Dec 2003 1.1
24 Oct 2003 1.0
"""

for l in CLANG_RELEASE_TABLE.splitlines():
    if not l:
        continue
    *date_parts, version = l.split()
    RELEASE_DATES["clang-" + version] = dateutil.parser.parse(" ".join(date_parts))

for k, v in list(RELEASE_DATES.items()):
    if k.count(".") == 1:
        RELEASE_DATES[k + ".0"] = v


def compiler_path_to_name(path):
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


NAMES_TO_PATHS = {}

for cc in GCCS + CLANGS:
    NAMES_TO_PATHS[compiler_path_to_name(cc)] = cc


ALL_COMPILERS = sorted(NAMES_TO_PATHS, key=RELEASE_DATES.__getitem__)

LATEST_COMPILER = ALL_COMPILERS[-1]
LATEST_GCC = [cc for cc in ALL_COMPILERS if "gcc-" in cc][-1]
LATEST_CLANG = [cc for cc in ALL_COMPILERS if "clang-" in cc][-1]


def compiler_results(source):
    result = []

    for cc in GCCS + CLANGS:
        name = compiler_path_to_name(cc)
        for opt in OPTIMISATION_LEVELS:
            result.append(
                {
                    "compiler": name,
                    "optimisation": opt,
                    "result": run_compiler(cc, opt, source),
                }
            )
    return result


def crashes_on(compiler, optimisation, crash_type):
    def accept(source):
        result = run_compiler(compiler, optimisation, source)
        if crash_type in result:
            return Classification.INTERESTING
        if "bad-source" in result:
            return Classification.INVALIDCHEAP
        return Classification.VALID

    accept.__name__ = "crashes_on(%r, %r, %r)" % (compiler, optimisation, crash_type)
    accept.__qualname__ = accept.__name__
    return accept


def check_validity(source):
    with tempfile.TemporaryDirectory() as d:
        sourcefile = os.path.join(d, "source.c")
        with open(sourcefile, "w") as o:
            o.write(source)

        env = dict(os.environ)
        env["LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"

        base_flags = ["-I", RUNTIME]

        def cc(compiler, *args, o=None):
            return subprocess.check_output(
                [
                    NAMES_TO_PATHS[compiler],
                    "-I",
                    RUNTIME,
                    *args,
                    sourcefile,
                    *(("-o", o) if o else ()),
                ],
                env=env,
                cwd=d,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
                encoding="utf-8",
            )

        try:
            gcc_errors = cc(LATEST_GCC, "-pedantic", "-Wall", "-O2", "-c", "-Wextra")
        except subprocess.CalledProcessError as e:
            return Classification.INVALIDCHEAP

        clang_bin = os.path.join(d, "testexe")
        try:
            clang_errors = cc(LATEST_CLANG, "-pedantic", "-Wall", "-O0", "-o/dev/null")
        except subprocess.CalledProcessError as e:
            # clang considers a few things to be errors that gcc does not.
            # e.g. if the second argument to main is not a char** this is an
            # error in clang but not in gcc. Additionally, we run the linking
            # step here.
            return Classification.INVALIDCHEAP

        if any(
            e in gcc_errors
            for e in [
                "uninitialized",
                "without a cast",
                "control reaches end",
                "return type defaults",
                "cast from pointer to integer",
                "useless type name in empty declaration",
                "no semicolon at end",
                "type defaults to",
                "too few arguments for format",
                "incompatible pointer",
                "ordered comparison of pointer with integer",
                "declaration does not declare anything",
                "expects type",
                "pointer from integer",
                "incompatible implicit",
                "excess elements in struct initializer",
                "comparison between pointer and integer",
            ]
        ):
            return Classification.INVALIDEXPENSIVE

        if any(
            e in clang_errors
            for e in [
                "conversions than data arguments",
                "incompatible redeclaration",
                "ordered comparison between pointer",
                "eliding middle term",
                "end of non-void function",
                "invalid in C99",
                "specifies type",
                "should return a value",
                "uninitialized",
                "incompatible pointer to",
                "incompatible integer to",
                "type specifier missing",
            ]
        ):
            return Classification.INVALIDEXPENSIVE

        for sanitizer in ["undefined"]:
            binary = os.path.join(d, "test." + sanitizer)

            try:
                subprocess.check_call(
                    [
                        NAMES_TO_PATHS[LATEST_CLANG],
                        "-I",
                        RUNTIME,
                        "-O1",
                        "-fsanitize=" + sanitizer,
                        sourcefile,
                        "-o",
                        binary,
                    ],
                    env=env,
                    cwd=d,
                    stderr=subprocess.STDOUT,
                    timeout=TIMEOUT,
                )

                assert os.path.exists(binary)

                subprocess.check_call(
                    [binary], env=env, cwd=d, timeout=TIMEOUT, stdout=subprocess.DEVNULL
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return Classification.INVALIDEXPENSIVE

        return Classification.VALID


def differs_from(reference, target, requires_validity_check=False):
    def accept(source):
        if requires_validity_check:
            c = check_validity(source)
            if c != Classification.VALID:
                return c
        desired = run_compiler(*reference, source)
        actual = run_compiler(*target, source)
        try:
            if desired["success"] != actual["success"]:
                return Classification.INTERESTING
        except KeyError:
            # Both of these *should* have succeeded, otherwise we would have
            # failed at check_validity. That means we've hit a different bug
            # here.
            return Classification.SLIPPAGE

        return Classification.VALID

    accept.__name__ = "differed_from(%r, %r)" % (reference, target)
    accept.__qualname__ = accept.__name__
    return accept


def error_predicate(info, check_validity=False):
    info = sorted(
        info,
        key=lambda x: (
            RELEASE_DATES[x["compiler"]],
            -OPTIMISATION_LEVELS.index(x["optimisation"]),
        ),
    )

    best_compiler = info[-1]
    desired_result = best_compiler["result"]["success"]

    for data in info:
        for c in ("compiler-crash", "bin-crash"):
            if c in data["result"]:
                return crashes_on(data["compiler"], data["optimisation"], c)
        else:
            try:
                if data["result"]["success"] != desired_result:
                    return differs_from(
                        (best_compiler["compiler"], best_compiler["optimisation"]),
                        (data["compiler"], data["optimisation"]),
                        requires_validity_check=check_validity,
                    )
            except KeyError:
                pass
    assert False, "No bug?"


def normalize_source(c_source):
    gcc = subprocess.Popen(
        [NAMES_TO_PATHS[LATEST_GCC], "-fpreprocessed", "-dD", "-P", "-E", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    fmt = subprocess.Popen(
        ["clang-format"], stdin=gcc.stdout, stdout=subprocess.PIPE, encoding="utf-8"
    )
    gcc.stdin.write(c_source)
    gcc.stdin.close()
    stdout, _ = fmt.communicate()
    for p in [fmt, gcc]:
        p.wait()
        if p.returncode != 0:
            raise Exception("%r exited with code %d" % (p.args, p.returncode))
    return stdout


define_experiment(
    "csmith",
    generator=csmith(),
    calculate_info=compiler_results,
    calculate_error_predicate=error_predicate,
    normalize_test_case=normalize_source,
    check_validity=check_validity,
)
