"""Public Inspect task entry points for LexRubric."""

from inspect_ai import Task, task

from lexrubric_inspect.lexrubric_port import JUDGE_MODEL
from lexrubric_inspect.lexrubric_port import lexrubric as _lexrubric


@task
def lexrubric(split: str = "all", judge_model: str = JUDGE_MODEL) -> Task:
    """Build the original LexRubric task with the selected split and judge."""
    return _lexrubric(split=split, judge_model=judge_model)
