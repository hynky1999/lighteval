from typing import get_args

from ..tasks.qa.mkqa import MkqaTask, TaskType

from ..tasks.mqa.exams import ExamsTask, subjects_by_lang_code
from ..tasks.utils.tasks_helpers import tasks_to_string

from ..tasks.qa.mintaka import MintakaTask
from ..tasks.nli.lambada import LambadaTask
from ..tasks.mqa.mlmm import get_mlmm_tasks, get_mmlu_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.nli.pawns import PawnsXTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.nli.xwinograd import XWinogradeTask
from ..tasks.suites.frenchbench import _GENERATIVE_TASKS as _FRENCH_BENCH_GENERATIVE_TASKS, _MC_TASKS as _FRENCH_BENCH_MC_TASKS
from ..tasks.mqa.meta_mmlu import MetaMMLUTask, MMLU_SUBSET

_MKQA_TASKS = [MkqaTask(lang="fr", type=type) for type in get_args(TaskType)]

_GENERATIVE_TASKS = [
    MintakaTask(lang="fr"),
    *_FRENCH_BENCH_GENERATIVE_TASKS,
    *_MKQA_TASKS
]

_MC_TASKS = [
    LambadaTask(lang="fr"),
    BelebeleTask(lang="fr"),
    PawnsXTask(lang="fr", version=1),
    PawnsXTask(lang="fr", version=2),
    XCODAHTask(lang="fr"),
    XCSQATask(lang="fr"),
    XNLITask(lang="fr", version=1),
    XNLITask(lang="fr", version=2),
    XWinogradeTask(lang="fr"),
    *get_mlmm_tasks("fr"),
    *_FRENCH_BENCH_MC_TASKS,
    *[MetaMMLUTask("fr", subset) for subset in get_args(MMLU_SUBSET)],
    *[ExamsTask(lang="fr", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["fr"] for show_options in [True, False]]
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="fr", version=version) for version in (1, 2)] +
                            [PawnsXTask(lang="fr", version=version) for version in (1, 2)]
                            ),
    "meta_mmlu": tasks_to_string([MetaMMLUTask("fr", subset) for subset in get_args(MMLU_SUBSET)]),
    "xcodah": tasks_to_string([XCODAHTask(lang="fr")]),
    "exams": tasks_to_string([ExamsTask(lang="fr", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["fr"] for show_options in [True, False]]),
    "mkqa": tasks_to_string(_MKQA_TASKS),
    "early-signal": tasks_to_string(['belebele-fr', 'fquadv2', 'french-hellaswag', *_MKQA_TASKS, 'mintaka-fr', *get_mmlu_tasks("fr"), 'pawns-v2-fr', 'x-codah-fr', 'x-csqa-fr', 'xnli-bool-v2-fr'])
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))