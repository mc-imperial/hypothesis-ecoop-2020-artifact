import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as nps
from evalcommon import eval_given

int16s = nps.from_dtype(np.dtype('int16'))


bounded_lists = st.lists(int16s, max_size=1).filter(lambda x: sum(x) < 256)


problems = st.tuples(
    bounded_lists,
    bounded_lists,
    bounded_lists,
    bounded_lists,
    bounded_lists,
)


if __name__ == '__main__':
    @eval_given(problems)
    def test(p):
        assert sum([x for sub in p for x in sub], np.int16(0)) < 5 * 256
