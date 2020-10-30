import hypothesis.strategies as st
from evalcommon import eval_given
from hypothesis import assume


if __name__ == '__main__':
    @eval_given(st.lists(st.integers(0, 10)))
    def test(ls):
        assume(all(v < len(ls) for v in ls))
        for i, j in enumerate(ls):
            if i != j:
                assert ls[j] != i
