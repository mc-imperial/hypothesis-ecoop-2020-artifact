import hypothesis.strategies as st
from evalcommon import eval_given


if __name__ == '__main__':
    @eval_given(st.lists(st.integers()))
    def test(ls):
        rev = list(reversed(ls))
        assert ls == rev
