from evalcommon import support_executable, eval_given
from subprocess import Popen, PIPE
import hypothesis.strategies as st
import string
from hypothesis.internal.conjecture.utils import Sampler
from hypothesis.strategies._internal.strategies import SearchStrategy


BUG = support_executable('parser', 'Bug')


def Var():
    return formatted(
        'Var "%s"',
        st.text(alphabet=string.ascii_letters + string.digits, min_size=1)
    )


def Mod():
    return formatted(
        "Mod { imports = %s, exports = %s }",
        string_lists(Var()),
        string_lists(Var()),
    )


def Func():
    return formatted(
        "Func{ fnName = %s, args = %s , stmts = %s }",
        Var(), string_lists(Exp()), string_lists(Stmt())
    )


def formatted(format_string, *args):
    return st.tuples(*args).map(lambda t: format_string % t)


def string_lists(*args, **kwargs):
    return st.lists(*args, **kwargs).map(
        lambda ls: "[%s]" % (', '.join(ls),)
    )


def Stmt():
    return st.one_of((
        formatted("Return (%s)", Exp()),
        formatted("Assign (%s) (%s)", Var(), Exp()),
        formatted("Alloc (%s) (%s)", Var(), Exp()),
    ))


class Frequency(SearchStrategy):
    def __init__(self, children):
        SearchStrategy.__init__(self)
        children = tuple(children)
        self.values = [
            t for _, t in children
        ]
        self.sampler = Sampler([s for s, _ in children])

    def do_draw(self, data):
        i = self.sampler.sample(data)
        return self.values[i]


EXP_PATTERNS = Frequency([
    (10,  "Not (%s)"),
    (100, "And (%s) (%s)"),
    (100, "Or (%s) (%s)"),
    (100, "Add (%s) (%s)"),
    (100, "Sub (%s) (%s)"),
    (100, "Mul (%s) (%s)"),
    (100, "Div (%s) (%s)"),
])


@st.composite
def Exp(draw, depth=None):
    if depth is None:
        depth = draw(st.integers(0, 100))

    if depth <= 0:
        return draw(st.one_of((
            formatted("Bool %s", st.booleans()),
            formatted("Int %s", st.integers()),
        )))

    child = Exp(depth=draw(st.integers(0, depth - 1)))
    pattern = draw(EXP_PATTERNS)

    if pattern.count("%s") == 1:
        return draw(formatted(pattern, child))
    else:
        return draw(formatted(pattern, child, child))


def Lang():
    return formatted(
        "Lang {modules=%s, funcs=%s}",
        string_lists(Mod()), string_lists(Func()),
    )


if __name__ == '__main__':
    @eval_given(Lang())
    def test(value):
        proc = Popen([BUG], stdin=PIPE, stdout=PIPE, encoding='utf-8')
        stdout, _ = proc.communicate(value)
        assert proc.returncode == 0
        assert stdout.strip() == 'True'
