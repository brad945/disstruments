"""Fine-grained instrument recognition: taxonomy, dataset loaders, eval harness.

Numpy + PyYAML only at import time; heavy deps (torch, audio decoding) stay out of this
package's core so `make test` runs in fake-ML mode.
"""
