# Distinct Elements in a List

This tests the example provided for the property "a list of integers containing
at least three distinct elements".

This is interesting because:

1. Most property-based testing libraries will not successfully normalize (i.e.
   always return the same answer) this property, because it requires reordering
   examples to do so.
2. Hypothesis and test.check both provide a built in generator for "a list of
   distinct elements", so the "example of size at least N" provides a sort of
   lower bound for how well they can shrink those built in generators.
