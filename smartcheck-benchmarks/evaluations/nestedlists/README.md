# Nested Lists

This tests the performance of shrinking a list of lists, subject to the
constraint that the sum of the element lists is at least 10.

The reason this is interesting is that it has lots of local minima under
pure deletion based approaches. e.g. `[[0], ..., [0]]` and `[[0, ..., 0]]` are
both minima for this under anything that can only make individual elements
smaller.

Hypothesis can shrink this reliably to `[[0, ..., 0]]` because it can merge two
adjacent lists by deleting the weighted boolean call that terminates the first
one together with the one that starts the second one as an element of the
parent list.

I [asked about getting some examples of where this might matter](https://twitter.com/DRMacIver/status/993829474696417280)
and Jacob Stanley (author of Hedgehog) gave me a nice answer:

> Yeah I have had this exact need recently. Generating packets that contain a
> number of independent messages, but ideally smaller means less packets in the
> stream. So merging packets is good. Ideally you end up with a single packet
> that has just the messages that cause the failure.
>
> I can’t provide the code unfortunately, but the packets in question here are
> market data following the ITCH protocol. So one other aspect is that messages
> are events so they must remain ordered even if they jump to another packet.

More generally this is interesting because it forms a kind of hyper-specific
boundary violating shrink that Hypothesis can do and basically nothing else
can. You'd need to write a custom shrinker that handled specifically lists of
lists (or maybe lists of monoids or something) in order to implement this.
