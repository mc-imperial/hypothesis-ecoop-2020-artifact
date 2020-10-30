# Deletion

This tests the property "if we remove an element from a list, the element is
no longer in the list". The remove function we use however only actually
removes the *first* instance of the element, so this fails whenever the list
contains a duplicate and we try to remove one of those elements.

This example is interesting for a couple of reasons:

1. It's a nice easy to explain example of property-based testing.
2. Shrinking duplicates simultaneously is something that most property-based
   testing libraries can't do.
3. I *thought* this was a relatively commonly used example, but actually I
   might be the originator of it. Currently hunting for a cite.
