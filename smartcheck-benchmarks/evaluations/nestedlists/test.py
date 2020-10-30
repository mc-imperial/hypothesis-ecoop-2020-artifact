import hypothesis.strategies as st
from evalcommon import eval_given


if __name__ == '__main__':
    @eval_given(st.lists(st.lists(st.just(()))))
    def test(ls):
        assert sum(map(len, ls), 0) <= 10
