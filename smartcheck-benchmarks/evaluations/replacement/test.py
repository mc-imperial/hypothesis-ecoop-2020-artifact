import hypothesis.strategies as st
from evalcommon import eval_given


if __name__ == '__main__':
    @eval_given(st.tuples(
        st.integers(0, 10 ** 6),
        st.lists(st.integers(2, 10)),
    ))
    def test(prob):
        n, pows = prob
        for p in pows:
            n *= p

        assert n < 10 ** 6
