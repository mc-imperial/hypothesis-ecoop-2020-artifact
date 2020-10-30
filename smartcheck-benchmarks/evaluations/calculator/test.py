import hypothesis.strategies as st
from hypothesis import assume
from evalcommon import eval_given


expression = st.deferred(lambda: st.one_of(
    st.integers(),
    st.tuples(st.just('+'), expression, expression),
    st.tuples(st.just('/'), expression, expression),
))


def div_subterms(e):
    if isinstance(e, int):
        return True
    if e[0] == '/' and e[-1] == 0:
        return False
    return div_subterms(e[1]) and div_subterms(e[2])


def evaluate(e):
    if isinstance(e, int):
        return e
    elif e[0] == '+':
        return evaluate(e[1]) + evaluate(e[2])
    else:
        assert e[0] == '/'
        return evaluate(e[1]) // evaluate(e[2])


if __name__ == '__main__':
    @eval_given(expression)
    def test(e):
        assume(div_subterms(e))
        evaluate(e)
