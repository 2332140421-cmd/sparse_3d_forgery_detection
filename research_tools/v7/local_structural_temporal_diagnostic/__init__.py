"""Offline joint diagnostic for the frozen V7 local structural pilot.

This package answers a bounded thesis question: whether the already saved
models, sparse observations, four-dimensional structural states and mean
aggregation are internally consistent.  Its immediate caller is the
diagnostic command, which reads frozen artifacts and writes an audit package;
it never trains, changes scores, or reruns a frontend.  A small dedicated
module is sufficient here; a framework, registry, or service would add no
evidence to this experiment.
"""

__all__: list[str] = []
