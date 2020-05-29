# Wrong Binary Heap

This is based on an example from QuickCheck's test suite (via the SmartCheck
paper). It generates binary heaps, and then uses a wrong implementation of
a function that converts the binary heap to a sorted list and asserts that the
result is sorted.

Interestingly Hypothesis (and I think SmartCheck and QuickCheck too) seems to
never find the smallest example here, which is the four valued heap `(0, None,
(0, (0, None, None), (1, None, None)))`. I think this is essentially because
small examples are "too sparse", so it's unable to find one by luck.
