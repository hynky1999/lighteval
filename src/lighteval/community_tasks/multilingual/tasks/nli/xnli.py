from typing import Literal

from lighteval.community_tasks.multilingual.tasks.utils.translation_literals import FULL_STOP, LANG_NAMES_INVERTED

from ..utils.prompts import get_xnli_prompt, fix_ending_punct
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]


class XNLITask(LightevalTaskConfig):
    def __init__(self, lang: LANGS, version: Literal[1,2]):
        super().__init__(
            name=f"xnli-bool{f'-v{version}' if version != 1 else ''}-{lang}",
            suite=("custom",),
            prompt_function=get_xnli_prompt(lang, version),
            hf_repo="facebook/xnli",
            hf_subset=lang,
            # XNLI does use normal dot for chinese
            filter=lambda x: fix_ending_punct(x["premise"], lang).endswith(FULL_STOP[lang]) and int(x["label"]) in [0, 2],
            evaluation_splits=("validation",),
            few_shots_split="train",
            few_shots_select=None,
            generation_size=-1,
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi, Metrics.prob_raw,  Metrics.prob_raw_norm, Metrics.prob_raw_norm_token,  Metrics.prob_raw_norm_pmi, 
            ),
        )


class XNLI2Task(LightevalTaskConfig):
    # Revised version of xnli
    def __init__(self, lang: LANGS, version: Literal[1,2]):
        super().__init__(
            name=f"xnli-2.0-bool{f'-v{version}' if version != 1 else ''}-{lang}",
            suite=("custom",),
            prompt_function=get_xnli_prompt(lang, version),
            hf_repo=f"Harsit/xnli2.0_{LANG_NAMES_INVERTED[lang]}",
            hf_subset="default",
            filter=lambda x: fix_ending_punct(x["premise"], lang).endswith(FULL_STOP[lang]) and int(x["label"]) in [0, 2],
            evaluation_splits=("test",),
            # TODO: Add the train set
            few_shots_select=None,
            generation_size=-1,
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi, Metrics.prob_raw,  Metrics.prob_raw_norm, Metrics.prob_raw_norm_token,  Metrics.prob_raw_norm_pmi, 
            ),
        )

class XNLIBasqueTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"xnli-basque",
            suite=("custom",),
            prompt_function=get_xnli_prompt("eu", 2),
            hf_repo=f"HiTZ/xnli-eu",
            hf_subset="eu_native",
            filter=lambda x: fix_ending_punct(x["premise"], "eu").endswith(FULL_STOP["eu"]) and int(x["label"]) in [0, 2],
            evaluation_splits=("test",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.prob_raw,  Metrics.prob_raw_norm, Metrics.prob_raw_norm_token, 
            ),
            )

class AfriXNLITask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"afric-xnli-sw",
            suite=("custom",),
            prompt_function=get_xnli_prompt("sw", 2),
            hf_repo=f"masakhane/afrixnli",
            hf_subset="swa",
            filter=lambda x: fix_ending_punct(x["premise"], "sw").endswith(FULL_STOP["sw"]) and int(x["label"]) in [0, 2],
            evaluation_splits=("test",),
            few_shots_split="validation",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.prob_raw,  Metrics.prob_raw_norm, Metrics.prob_raw_norm_token, 
            ),
            )