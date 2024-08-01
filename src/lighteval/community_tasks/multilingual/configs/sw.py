from ..tasks.utils.tasks_helpers import tasks_to_string
from ..tasks.mqa.xcopa import XCopaTask
from ..tasks.mqa_with_context.m3exam import M3ExamTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.suites.swahili_leaderboard import TASKS as SW_TASKS
from ..tasks.qa.tydiqa import TydiqaTask


_GENERATIVE_TASKS = [
    TydiqaTask(lang="sw"),
]

_MC_TASKS = [
    BelebeleTask(lang="sw"),
    XStoryClozeTask(lang="sw"),
    XCopaTask(lang="sw"),
    XNLITask(lang="sw", version=1),
    XNLITask(lang="sw", version=2),
    M3ExamTask(lang="sw"),
    XCSQATask(lang="sw"),
    XCODAHTask(lang="sw"),
    *SW_TASKS
]

_ALL_TASKS = _GENERATIVE_TASKS + _MC_TASKS

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="sw", version=version) for version in (1, 2)]),
    "xcodah": tasks_to_string([XCODAHTask(lang="sw")]),
    "early-signal": tasks_to_string(['arc-sw:easy', 'belebele-sw', 'm3exam-sw', 'x-csqa-sw', 'xcopa-sw', 'xnli-bool-v2-sw', 'xstory_cloze-sw'])
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))