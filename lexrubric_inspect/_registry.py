"""Inspect AI extension entry point."""

from inspect_ai import task
from inspect_ai._util.registry import registry_info

from .lexrubric_port import lexrubric

# A source-tree .egg-info can hide editable-install metadata from Inspect's
# namespace detection. Keep the public task name stable in that situation.
if registry_info(lexrubric).name == "lexrubric":
    lexrubric = task(name="lexrubric_inspect/lexrubric")(lexrubric)

__all__ = ["lexrubric"]
