# Evaluation examples

This is a set of examples that are designed for comparing the shrinking capabilities of
different property based testing libraries.

## The Plan

The idea is to eventually have examples of each evaluation for each of the following
four libraries:

1. Hypothesis
2. test.check
3. QuickCheck
4. SmartCheck

We will compare these on two questions:

1. What is the minimum size of the example found? (We will need to come up with some consistent notion of size in some cases,
   but where possible we should follow the SmartCheck paper's lead).
2. Does the example normalize, in the sense that the library returns the same example (with high probability) each time it is
   run.

These examples come from two main sources:

1. The paper "Pike, Lee. "SmartCheck: automatic and efficient counterexample reduction and generalization." ACM SIGPLAN Notices. Vol. 49. No. 12. ACM, 2014.",
   which has a number of examples it uses for comparison of its reduction to QuickCheck.
2. Examples which demonstrate Hypothesis doing better than other libraries, usually extracted from Hypothesis's test suite.

We can definitely add more examples if other interesting sources come up.

## TODO

* We are still missing the parser/pretty-printer example from the SmartCheck paper, mostly because it will be annoying to port reasonably faithfully to different languages.
