
from typing import get_args

from ..tasks.mqa.xcopa import XCopaTaskEU

from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask

from ..tasks.qa.custom_squad import BasqueSquad

from ..tasks.qa.mkqa import MkqaTask, TaskType

from ..tasks.mqa.exams import ExamsTask, subjects_by_lang_code
from ..tasks.utils.tasks_helpers import tasks_to_string

from ..tasks.qa.mintaka import MintakaTask
from ..tasks.nli.lambada import LambadaTask
from ..tasks.mqa.mlmm import get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.nli.pawns import PawnsXTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLI2Task, XNLIBasqueTask, XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.suites.frenchbench import _GENERATIVE_TASKS as _FRENCH_BENCH_GENERATIVE_TASKS, _MC_TASKS as _FRENCH_BENCH_MC_TASKS
from ..tasks.suites.eus_evals import BasqueRC, BasqueReadingProficiency, BasqueTrivia, BasqueExams, EXAMS_SUBSETS, BertaQATask


_GENERATIVE_TASKS = [
    BasqueSquad(lang="eu"),
]

_MC_TASKS = [
    XNLIBasqueTask(),
    BertaQATask(lang="eu"),
    *get_mlmm_tasks("eu"),
    XStoryClozeTask(lang="eu"),
    BelebeleTask(lang="eu"),
    XCopaTaskEU(),
    BasqueRC(),
    BasqueTrivia(),
    *[BasqueExams(subset=subset) for subset in EXAMS_SUBSETS],
    BasqueReadingProficiency()
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))


TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))