from typing import get_args

from ..tasks.utils.tasks_helpers import tasks_to_string
from ..tasks.mqa.exams import ExamsTask, subjects_by_lang_code
from ..tasks.qa.tquad import Tquad2Task
from ..tasks.suites.turkish_leaderboard import ARCEasyTrTask, HellaSwagTrTask, MMLUTaskTr, TruthfulQATrTask, WinogradeTrTask, MMLU_SUBSETS
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.nli.xnli import XNLITask
from ..tasks.mqa.xcopa import XCopaTask
from ..tasks.qa.mkqa import MkqaTask, TaskType

_MKQA_TASKS = [MkqaTask(lang="tr", type=task_type) for task_type in get_args(TaskType)]

_GENERATIVE_TASKS = [
    XquadTask(lang="tr"),
    Tquad2Task(),
    *_MKQA_TASKS
]

_MC_TASKS = [
    BelebeleTask(lang="tr"),
    XNLITask(lang="tr", version=1),
    XNLITask(lang="tr", version=2),
    XCopaTask(lang="tr"),
    HellaSwagTrTask(),
    TruthfulQATrTask("mc1"),
    TruthfulQATrTask("mc2"),
    ARCEasyTrTask(version=2),
    WinogradeTrTask(),
    *[MMLUTaskTr(subset) for subset in get_args(MMLU_SUBSETS)],
    *[ExamsTask(lang="tr", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["tr"] for show_options in [True, False]]
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))
TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="tr", version=version) for version in (1, 2)]),
    "arc": tasks_to_string([ARCEasyTrTask(version=2)]),
    "exams": tasks_to_string([ExamsTask(lang="tr", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["tr"] for show_options in [True, False]]),
    "mkqa": tasks_to_string(_MKQA_TASKS),
    "early-signal": tasks_to_string(['arc-v2-tr', 'belebele-tr', 'exams-tr:Biology', 'exams-tr:Business', 'exams-tr:History', 'exams-tr:Philosophy', 'hellaswag-tr', *_MKQA_TASKS, *[MMLUTaskTr(subset) for subset in get_args(MMLU_SUBSETS)], 'tqduad2', 'xcopa-tr', 'xnli-bool-v2-tr', 'xquad-tr'])
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))