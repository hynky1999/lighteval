from ..tasks.utils.tasks_helpers import tasks_to_string
from ..tasks.qa.Indicqa import IndicQATask
from ..tasks.nli.indicnxnli import XNLIIndicTask
from ..tasks.mqa.indicxcopa import XCopaIndicTask
from ..tasks.mqa.mlmm import get_mlmm_tasks, get_mmlu_tasks
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.qa.tydiqa import TydiqaTask


_GENERATIVE_TASKS = [
    TydiqaTask(lang="te"),
    IndicQATask(lang="te"),
]

_MC_TASKS = [
    XNLIIndicTask(lang="te", version=1),
    XNLIIndicTask(lang="te", version=2),
    XCopaIndicTask(lang="te"),
    BelebeleTask(lang="te"),
    XStoryClozeTask(lang="te"),
    *get_mlmm_tasks("te")
]

_ALL_TASKS = list(set(_GENERATIVE_TASKS + _MC_TASKS))

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLIIndicTask(lang="te", version=version) for version in (1, 2)]),
    "early-signal": tasks_to_string(['belebele-te', 'hellaswag-te', 'indicqa.te', *get_mmlu_tasks("te"), 'tydiqa-te', 'xcopa-te', 'xstory_cloze-te'])
}

TASKS_TABLE = [task.as_dict() for task in _GENERATIVE_TASKS + _MC_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
