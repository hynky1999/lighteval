from typing import get_args
from lighteval.community_tasks.multilingual.tasks.mqa.afric_mmlu import AfricMMLUTask
from ..tasks.qa.mkqa import MkqaTask, TaskType
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.mqa.exams import ExamsTask, subjects_by_lang_code
from ..tasks.qa.mlqa import MlqaTask
from ..tasks.utils.tasks_helpers import tasks_to_string
from ..tasks.qa.mintaka import MintakaTask
from ..tasks.mqa.mlmm import M_MMLUTask, get_mlmm_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.nli.pawns import PawnsXTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLI2Task, XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.suites.frenchbench import _GENERATIVE_TASKS as _FRENCH_BENCH_GENERATIVE_TASKS, _MC_TASKS as _FRENCH_BENCH_MC_TASKS
from ..tasks.mqa.meta_mmlu import MetaMMLUTask, MMLU_SUBSET

_GENERATIVE_TASKS = [
    XquadTask(lang="de"),
    MlqaTask(lang="de"),
    MintakaTask(lang="de"),
]

_MC_TASKS = [
    BelebeleTask(lang="de"),
    XCODAHTask(lang="de"),
    XCSQATask(lang="de"),
    PawnsXTask(lang="de", version=2),
    XNLI2Task(lang="de", version=2),
    *get_mlmm_tasks("de"),
    *[MetaMMLUTask("de", subset) for subset in get_args(MMLU_SUBSET)],
    *[ExamsTask(lang="de", subject=subject, show_options=False) for subject in subjects_by_lang_code["de"]],
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