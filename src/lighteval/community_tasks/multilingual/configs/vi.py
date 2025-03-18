from typing import get_args

from lighteval.community_tasks.multilingual.tasks.mqa.afric_mmlu import AfricMMLUTask
from lighteval.community_tasks.multilingual.tasks.mqa.xcopa import XCopaTask
from lighteval.community_tasks.multilingual.tasks.mqa_with_context.m3exam import M3ExamTask
from lighteval.community_tasks.multilingual.tasks.qa.mlqa import MlqaTask

from ..tasks.qa.mkqa import MkqaTask, TaskType

from ..tasks.mqa.exams import ExamsTask, subjects_by_lang_code
from ..tasks.utils.tasks_helpers import tasks_to_string

from ..tasks.qa.mintaka import MintakaTask
from ..tasks.nli.lambada import LambadaTask
from ..tasks.mqa.mlmm import M_MMLUTask, get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.nli.pawns import PawnsXTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLI2Task, XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.suites.frenchbench import _GENERATIVE_TASKS as _FRENCH_BENCH_GENERATIVE_TASKS, _MC_TASKS as _FRENCH_BENCH_MC_TASKS
from ..tasks.mqa.meta_mmlu import MetaMMLUTask, MMLU_SUBSET
from ..tasks.mqa_with_context.xquad import XquadTask

_GENERATIVE_TASKS = [
    MlqaTask(lang="vi"),
    XquadTask(lang="vi"),
]

_MC_TASKS = [
    BelebeleTask(lang="vi"),
    XCODAHTask(lang="vi"),
    XCSQATask(lang="vi"),
    XCopaTask(lang="vi"),
    XNLI2Task(lang="vi", version=2),
    M3ExamTask(lang="vi", version=1),
    *[ExamsTask(lang="vi", subject=subject, show_options=False) for subject in subjects_by_lang_code["vi"]],
    *get_mlmm_tasks("vi"),
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