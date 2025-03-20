from typing import get_args

from lighteval.community_tasks.multilingual.tasks.mqa_with_context.m3exam import M3ExamTask
from lighteval.community_tasks.multilingual.tasks.qa.custom_squad import ItalianSQuADTask

from ..tasks.qa.mkqa import MkqaTask, TaskType

from ..tasks.mqa.exams import ExamsTask, subjects_by_lang_code
from ..tasks.utils.tasks_helpers import tasks_to_string

from ..tasks.qa.mintaka import MintakaTask
from ..tasks.nli.lambada import LambadaTask
from ..tasks.mqa.mlmm import M_MMLUTask, get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa.xcopa import XCopaTask
from ..tasks.nli.pawns import PawnsXTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLI2Task, XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.suites.frenchbench import _GENERATIVE_TASKS as _FRENCH_BENCH_GENERATIVE_TASKS, _MC_TASKS as _FRENCH_BENCH_MC_TASKS
from ..tasks.mqa.meta_mmlu import MetaMMLUTask, MMLU_SUBSET

_GENERATIVE_TASKS = [
    MintakaTask(lang="it"),
    ItalianSQuADTask(),
    *[MkqaTask(lang="it", type=type) for type in get_args(TaskType)]
]

_MC_TASKS = [
    # LambadaTask(lang="fr"),
    BelebeleTask(lang="it"),
    XCopaTask(lang="it"),
    XCODAHTask(lang="it"),
    XCSQATask(lang="it"),
    *get_mlmm_tasks("it"),
    *[ExamsTask(lang="it", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["it"] for show_options in [True, False]],
    M3ExamTask(lang="it", version=1),
    *[MetaMMLUTask("it", subset) for subset in get_args(MMLU_SUBSET)],
    *[ExamsTask(lang="it", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["it"] for show_options in [True, False]]
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "early-signals": tasks_to_string(_MC_TASKS + _GENERATIVE_TASKS),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))