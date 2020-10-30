# Reversing a list

This tests the (wrong) property that reversing a list results in the same list.

This is mostly here because it's in the SmartCheck paper, but there's a decent
chance of it failing to normalize reliably in most libraries - it's essentially
looking for two distinct elements, so shares some overlap with the `distinct`
evaluation example.
