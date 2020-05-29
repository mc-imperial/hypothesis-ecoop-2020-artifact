# SmartCheck's "motivating example"

The SmartCheck paper starts with an example as follows: Given a 5-tuple of
lists of 16-bit integers, we want to test the property that if each list sums
to less than 256, then the sum of all the values in the lists is less than
5 * 256. This is false because of overflow. e.g.
`([-20000], [-20000], [], [], [])` is a counter-example.

This example is designed to trigger pathological shrinking behaviour in
QuickCheck's default shrink implementation - the starting examples are large,
and QuickCheck ends up attempting over 10^10 shrinks (I'm a little suspicious
of this claim and think we should validate it - although true in principle I
think laziness of shrinking means that in practice it ends up pretty efficient).
