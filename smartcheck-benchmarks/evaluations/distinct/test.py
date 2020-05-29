import hypothesis.strategies as st
from evalcommon import eval_given

if __name__ == '__main__':
    @eval_given(st.lists(st.integers()))
    def test_list_has_few_distinct(ls):
        assert len(set(ls)) < 3
