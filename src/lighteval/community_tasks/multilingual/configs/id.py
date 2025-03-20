from typing import get_args

from lighteval.community_tasks.multilingual.tasks.mqa.afric_mmlu import AfricMMLUTask
from lighteval.community_tasks.multilingual.tasks.mqa.xcopa import XCopaTask
from lighteval.community_tasks.multilingual.tasks.qa.tydiqa import TydiqaTask

from ..tasks.mqa.indommlu import IMMLU_TASK_TYPE
from ..tasks.qa.mkqa import MkqaTask, TaskType

from ..tasks.mqa.exams import ExamsTask, subjects_by_lang_code
from ..tasks.utils.tasks_helpers import tasks_to_string

from ..tasks.qa.mintaka import MintakaTask
from ..tasks.mqa.indommlu import IndonesianMMLUTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.lambada import LambadaTask
from ..tasks.mqa.mlmm import M_MMLUTask, get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.nli.pawns import PawnsXTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLI2Task, XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.suites.frenchbench import _GENERATIVE_TASKS as _FRENCH_BENCH_GENERATIVE_TASKS, _MC_TASKS as _FRENCH_BENCH_MC_TASKS
from ..tasks.mqa.meta_mmlu import MetaMMLUTask, MMLU_SUBSET

_GENERATIVE_TASKS = [
    TydiqaTask(lang="id"),
]

_MC_TASKS = [
    BelebeleTask(lang="id"),
    XCopaTask(lang="id"),
    XStoryClozeTask(lang="id"),
    *[IndonesianMMLUTask(task) for task in get_args(IMMLU_TASK_TYPE)],
    *get_mlmm_tasks("id"),
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))


TASKS_GROUPS = {
    "early-signals": tasks_to_string(_MC_TASKS + _GENERATIVE_TASKS),
    "all": tasks_to_string(_ALL_TASKS),
    "mmlu": tasks_to_string([task for task in _ALL_TASKS if isinstance(task, IndonesianMMLUTask)]),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))