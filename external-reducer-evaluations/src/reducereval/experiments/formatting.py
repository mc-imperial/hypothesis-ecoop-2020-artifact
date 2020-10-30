import hypothesis.strategies as st
from hypothesis import (
    assume,
    given,
    settings,
    Verbosity,
    HealthCheck,
    note,
    unlimited,
    Phase,
)
import operator
import math
import inspect
from contextlib import contextmanager
import tempfile
import os
import subprocess
import re
from hypothesis.internal.conjecture.engine import (
    ConjectureRunner,
    Status,
    StopTest,
    ConjectureData,
    RunIsComplete,
    sort_key,
)
import hypothesis.internal.conjecture.engine as eng
from random import Random
from hypothesis.errors import UnsatisfiedAssumption
import hashlib
from hypothesis.searchstrategy import SearchStrategy
import attr
import json
import gzip
import base64
import sys
import time
from reducereval.experiments.experiments import define_experiment
from reducereval.reduction import Classification
import ast


VERSION = 2


@contextmanager
def string_as_python_file(contents):
    try:
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "contents.py")
            with open(f, "w") as o:
                o.write(contents)
            yield f
    finally:
        try:
            pass  # os.unlink(f)
        except FileNotFoundError:
            pass


def cached(f):
    cache = {}

    def accept(*args):
        try:
            return cache[args]
        except KeyError:
            return cache.setdefault(args, f(*args))

    return accept


def is_valid(s):
    try:
        compile(s, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def black(s):
    return subprocess.check_output(
        ["black", "-", "-l80"], stderr=subprocess.DEVNULL, input=s, encoding="utf-8"
    )


def yapf(s):
    with string_as_python_file(s) as t:
        subprocess.check_call(
            ["yapf", t, "-i", "--style=pep8"],
            # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        with open(t) as i:
            return i.read()


ERROR_MATCHER = re.compile(r"[^\s]+ ([A-Z][0-9]*) .*")


@cached
def pycodestyle_errors(s):
    with string_as_python_file(s) as t:
        output = subprocess.run(
            ["pycodestyle", t], check=False, stdout=subprocess.PIPE, encoding="utf-8"
        ).stdout
        result = set()
        for l in output.splitlines():
            m = ERROR_MATCHER.match(l)
            if m is not None:
                result.add(m[1])
        return frozenset(result)


identifier = st.from_regex(r"\A[a-zA-Z_][a-zA-Z_]*\Z").filter(is_valid)


def comp(f):
    return st.deferred(st.composite(f))


@comp
def argspec(draw):
    arguments = draw(st.lists(identifier, unique=True))
    varargs = draw(st.booleans())
    kwargs = draw(st.booleans())

    assume(varargs + kwargs <= len(arguments))
    n = 1
    if kwargs:
        arguments[-n] = "**" + arguments[-n]
        n += 1
    if varargs:
        arguments[-n] = "*" + arguments[-n]
    return ", ".join(arguments)


def several(s):
    return st.builds(", ".join, st.lists(s))


@st.deferred
def naturally_unambiguous_expression():
    def float_repr(s):
        if math.isinf(s) or math.isnan(s):
            return "float(%r)" % (repr(s),)
        else:
            return repr(s)

    return st.one_of(
        [s.map(repr) for s in [st.integers(), st.floats(), st.binary()]]
        + [
            identifier,
            st.floats().map(float_repr),
            # bracket(st.builds(', '.join, st.lists(unambiguous_expression, max_size=2))),
            formatted("%s(%s)", identifier, several(unambiguous_expression)),
        ]
    )


def formatted(s, *args):
    return st.tuples(*args).map(lambda t: s % t)


@st.deferred
def lambda_expression():
    return formatted("lambda %s: %s", argspec, unambiguous_expression)


@st.deferred
def unambiguous_expression():
    return naturally_unambiguous_expression | bracket(expression)


@st.deferred
def unary_operator_expression():
    return formatted(
        "%s %s", st.sampled_from(("not", "-", "+", "~")), unambiguous_expression
    )


@comp
def binary_operator_expression(draw):
    operators = ["+", "-", "*", "**", "%", "/", "//", "or", "and"]
    parts = [draw(unambiguous_expression)]
    for t in draw(
        st.lists(
            st.tuples(st.sampled_from(operators), unambiguous_expression), min_size=1
        )
    ):
        parts.extend(t)
    return " ".join(parts)


@st.deferred
def expression():
    if VERSION == 1:
        return naturally_unambiguous_expression | lambda_expression
    else:
        return (
            naturally_unambiguous_expression
            | lambda_expression
            | unary_operator_expression
            | binary_operator_expression
        )


def bracket(exps):
    return st.builds(operator.mod, st.sampled_from(("(%s)", "[%s]", "{%s}")), exps)


@st.deferred
def unambiguous_expression():
    return naturally_unambiguous_expression | bracket(lambda_expression)


def indent(s):
    return "\n".join(["    " + l for l in s.splitlines()])


@st.deferred
def assignment():
    return formatted("%s = %s", identifier, expression)


@comp
def body(draw):
    level = draw(current_level)

    non_trivial = draw(st.booleans())

    if level[0] >= 3 or not non_trivial:
        return "    pass"
    else:
        try:
            level[0] += 1
            return draw(
                st.lists(statement, min_size=1).map(lambda ls: indent("\n".join(ls)))
            )
        finally:
            level[0] -= 1


@st.deferred
def function_definition():
    return formatted("def %s(%s):\n%s", identifier, argspec, body)


@st.deferred
def class_definition():
    return formatted("class %s(%s):\n%s", identifier, several(identifier), body)


@comp
def if_statement(draw):
    conditionals = draw(st.lists(st.tuples(unambiguous_expression, body), min_size=1))

    parts = []
    for condition, b in conditionals:
        parts.append("if %s:" % (condition,))
        parts.append(b)

    if draw(st.booleans()):
        parts.append("else:")
        parts.append(draw(body))

    return "\n".join(parts)


@comp
def while_statement(draw):
    parts = []
    parts.append("while %s:" % (draw(unambiguous_expression),))
    parts.append(draw(body))

    if draw(st.booleans()):
        parts.append("else:")
        parts.append(draw(body))

    return "\n".join(parts)


@comp
def try_statement(draw):
    try_body = draw(body)

    parts = ["try:", try_body]

    has_except = False

    for e, except_body in draw(st.lists(st.tuples(expression, body))):
        has_except = True
        parts.append("except %s:" % (e,))
        parts.append(indent(except_body))

    if draw(st.booleans()):
        has_except = True
        parts.append("except:")
        parts.append(indent(draw(body)))

    has_else = draw(st.booleans())

    assume(has_except or not has_else)

    if has_else:
        parts.append("else:")
        parts.append(indent(draw(body)))

    has_finally = draw(st.booleans())

    assume(has_finally or has_except)

    if has_finally:
        parts.append("finally:")
        parts.append(indent(draw(body)))

    return "\n".join(parts)


@st.deferred
def small_statement():
    return st.one_of(st.just("pass"), expression, assignment)


@st.deferred
def composite_statement():
    return st.one_of(
        if_statement,
        while_statement,
        try_statement,
        function_definition,
        class_definition,
    ).filter(lambda s: "\n" + "    " * 4 not in s)


@st.deferred
def statement():
    return st.one_of(small_statement, composite_statement)


current_level = st.shared(st.builds(lambda: [0]), key="shared level")


@comp
def python(draw):
    interesting = draw(st.lists(composite_statement, min_size=1))
    boring = draw(st.lists(statement, min_size=0))
    padding_lines = [""] * draw(st.integers(0, 100))
    return (
        "\n".join(draw(st.permutations(interesting + boring + padding_lines))).strip()
        + "\n"
    )


def calculate_info(python_source):
    black_formatted = black(python_source)
    yapf_formatted = yapf(black_formatted)
    error, = pycodestyle_errors(yapf_formatted)
    return {"black": black_formatted, "yapf": yapf_formatted, "error": error}


def has_error(error):
    def accept(source):
        try:
            black_formatted = black(source)
        except subprocess.SubprocessError:
            return Classification.INVALIDCHEAP
        try:
            yapf_formatted = yapf(black_formatted)
        except subprocess.SubprocessError:
            # We triggered a yapf crash, which is not the bug we were looking
            # for.
            return Classification.SLIPPAGE
        errors = pycodestyle_errors(yapf_formatted)
        if list(errors) == [error]:
            return Classification.INTERESTING
        else:
            return Classification.VALID

    return accept


def check_validity(source):
    try:
        ast.parse(source)
        return Classification.VALID
    except SyntaxError:
        return Classification.INVALIDCHEAP


define_experiment(
    "formatting",
    generator=statement,
    calculate_info=calculate_info,
    calculate_error_predicate=lambda info, check_validity=False: has_error(
        info["error"]
    ),
    normalize_test_case=black,
    check_validity=check_validity,
)
