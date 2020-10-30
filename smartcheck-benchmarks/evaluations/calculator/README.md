# Calculator

This example is based on the "simple calculator language" example from the
SmartCheck paper.

The idea is that we have a simple calculator language (expressed as an ADT in
Haskell) representing expressions like 1 + (2 / 3). The property being tested
is that if we have no subterms of the form x / 0, then we can evaluate the
expression without a zero division error.

This property is false, because we might have a term like 1 / (3 + -3).
