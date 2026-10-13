# piplab.datasets: acquisition and characterisation.
#
# Sources restricted to UCI and OpenML per Chapter 1 section 1.7.4.
# Characteristics are copied onto every run row so trend queries need
# no join.
from piplab.datasets.loaders import (
    CACHE_DIR,
    REGISTRY,
    DatasetError,
    DatasetSpec,
    characterise,
    load_dataset,
    offline_names,
    registered_names,
)

__all__ = [
    "CACHE_DIR", "REGISTRY", "DatasetError", "DatasetSpec",
    "characterise", "load_dataset", "offline_names", "registered_names",
]
