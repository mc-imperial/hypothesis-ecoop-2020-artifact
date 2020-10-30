import hypothesis.strategies as st
from hypothesis import assume
from evalcommon import eval_given

if __name__ == '__main__':
    @eval_given(st.tuples(st.lists(st.integers()), st.integers(0, 10)))
    def test(lsi):
        ls, i = lsi
        assume(i < len(ls))
        x = ls[i]
        ls.remove(x)
        assert x not in ls
