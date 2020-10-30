import hypothesis.strategies as st
from evalcommon import eval_given


# A heap is either None or a triple (x, h1, h2) with x <= every element of h1,
# h2.


@st.composite
def heap1(draw, bound, size):
    branch = draw(st.integers(1, 8))

    if branch == 1 or size <= 0:
        return None

    key = st.integers()

    # Note: This is a slight cheat. The idiomatic way of doing this would be
    # to use min_value=bound. Doing it this way lets us avoid some shrinking
    # bugs in the implementation of min_value (not the shrinker itself) which
    # I don't want to fix until we've finished because it changes the data
    # format.
    if bound is not None:
        key = key.filter(lambda n: n >= bound)

    head = draw(key)

    child_size = size // 2

    return (
        head,
        draw(heap1(head, child_size)),
        draw(heap1(head, child_size)),
    )


@st.composite
def heap(draw):
    return draw(heap1(None, draw(st.integers(0, 20))))


def to_list(heap):
    result = []
    stack = [heap]
    while stack:
        h = stack.pop()
        if h is None:
            continue
        result.append(h[0])
        stack.extend(h[1:])
    return result


def merge_heaps(h1, h2):
    if h1 is None:
        return h2
    if h2 is None:
        return h1
    if h1[0] <= h2[0]:
        return (h1[0], merge_heaps(h1[2], h2), h1[1])
    else:
        return (h2[0], merge_heaps(h2[2], h1), h2[1])


def to_sorted_list(heap):
    result = []
    while heap is not None:
        result.append(heap[0])
        heap = merge_heaps(heap[1], heap[2])
    return result


def wrong_to_sorted_list(heap):
    if heap is None:
        return []
    else:
        return [heap[0]] + to_list(merge_heaps(heap[1], heap[2]))


if __name__ == '__main__':
    @eval_given(heap())
    def test_result_is_sorted(h):
        l1 = to_list(h)
        l2 = wrong_to_sorted_list(h)

        assert l2 == sorted(l2)
        assert sorted(l1) == l2
