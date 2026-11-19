"""Inspect AI extension entry point."""

from inspect_ai import task
from inspect_ai.util import registry_info

from .tasks import lexrubric

# A source-tree .egg-info can hide editable-install metadata from Inspect's
# namespace detection. Keep the public task name stable in that situation.
if registry_info(lexrubric).name == "lexrubric":
    lexrubric = task(name="lexrubric_inspect/lexrubric")(lexrubric)

__all__ = ["lexrubric"]
