from typing import get_args

from ..tasks.mqa.arabic_mmlu import AR_MMLU_TASK_TYPE, ArabicMMLUTask
from ..tasks.qa.mkqa import MkqaTask, TaskType
from ..tasks.mqa.exams import ExamsTask, subjects_by_lang_code
from ..tasks.mqa_with_context.belebele import BelebeleTask
from ..tasks.utils.tasks_helpers import tasks_to_string
from ..tasks.mqa.xcopa import XCopaTask
from ..tasks.mqa.mlmm import get_mlmm_tasks
from ..tasks.mqa_with_context.xquad import XquadTask
from ..tasks.mqa_with_context.xstory_cloze import XStoryClozeTask
from ..tasks.nli.xcsr import XCODAHTask, XCSQATask
from ..tasks.nli.xnli import XNLITask
from ..tasks.qa.mlqa import MlqaTask
from ..tasks.qa.tydiqa import TydiqaTask
from ..tasks.suites.arabic_evals import GENERATIVE_TASKS as ARABIC_EVALS_GENERATIVE_TASKS, MC_TASKS as ARABIC_EVALS_MC_TASKS


_GENERATIVE_TASKS = [
    MlqaTask(lang="ar"),
    TydiqaTask(lang="ar"),
    XquadTask(lang="ar"),
    BelebeleTask(lang="ar"),
    *ARABIC_EVALS_GENERATIVE_TASKS,
    *[MkqaTask(lang="ar", type=task_type) for task_type in get_args(TaskType)],
]

_NATIVE_MMLU_TASKS = [ArabicMMLUTask(task=task) for task in get_args(AR_MMLU_TASK_TYPE)]

_MC_TASKS = [
    XCODAHTask(lang="ar"),
    XCopaTask(lang="ar"),
    XCSQATask(lang="ar"),
    XNLITask(lang="ar", version=1),
    XNLITask(lang="ar", version=2),
    XStoryClozeTask(lang="ar"),
    *get_mlmm_tasks("ar"),
    *ARABIC_EVALS_MC_TASKS,
    *[ExamsTask(lang="ar", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["ar"] for show_options in [True, False]],
    *_NATIVE_MMLU_TASKS
]

_ALL_TASKS = _GENERATIVE_TASKS + _MC_TASKS

TASKS_GROUPS = {
    "all": tasks_to_string(_ALL_TASKS),
    "generative": tasks_to_string(_GENERATIVE_TASKS),
    "mc": tasks_to_string(_MC_TASKS),
    "xnli": tasks_to_string([XNLITask(lang="ar", version=version) for version in (1, 2)]),
    "belebele": tasks_to_string([BelebeleTask(lang="ar")]),
    "exams": tasks_to_string([ExamsTask(lang="ar", subject=subject, show_options=show_options) for subject in subjects_by_lang_code["ar"] for show_options in [True, False]]),
    "xcodah": tasks_to_string([XCopaTask(lang="ar")]),
    "mkqa": tasks_to_string([MkqaTask(lang="ar", type=task_type) for task_type in get_args(TaskType)]),
    "arabic_mmlu": tasks_to_string(_NATIVE_MMLU_TASKS),
    "early-signal": tasks_to_string(['alghafa:mcq_exams_test_ar', 'alghafa:meta_ar_msa', 'alghafa:multiple_choice_grounded_statement_soqal_task', 'alghafa:multiple_choice_grounded_statement_xglue_mlqa_task', 'arabic_exams', *_NATIVE_MMLU_TASKS, 'arc_easy_ar', *[f"{'exams-ar':{subset}}" for subset in subjects_by_lang_code["ar"]], 'hellaswag-ar', 'mlqa-ar', 'piqa_ar', 'race_ar', 'sciq_ar', 'tydiqa-ar', 'x-codah-ar', 'x-csqa-ar', 'xnli-bool-v2-ar', 'xquad-ar', 'xstory_cloze-ar'])
}

TASKS_TABLE = [task.as_dict() for task in _ALL_TASKS]

if __name__ == "__main__":
    print([t for t in TASKS_TABLE])
    print(len(TASKS_TABLE))