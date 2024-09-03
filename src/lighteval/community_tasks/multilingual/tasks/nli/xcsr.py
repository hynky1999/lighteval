from typing import Literal

from ..utils.prompts import get_m_xcsr_prompt, get_xcodah_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pl", "pt", "ru", "sw", "vi", "zh"]


class XCODAHTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.lang = lang
        super().__init__(
            name=f"x-codah-{lang}",
            prompt_function=get_xcodah_prompt(lang),
            suite=("custom",),
            hf_repo="INK-USC/xcsr",
            hf_subset=f"X-CODAH-{lang}",
            evaluation_splits=("validation",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace, Metrics.loglikelihood_acc_norm_token, Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi,
                    Metrics.prob_raw, Metrics.prob_raw_norm, Metrics.prob_raw_norm_token, Metrics.prob_raw_norm_pmi],
        )


class XCSQATask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        self.lang = lang
        super().__init__(
            name=f"x-csqa-{lang}",
            prompt_function=get_m_xcsr_prompt(lang),
            suite=("custom",),
            hf_repo="INK-USC/xcsr",
            hf_subset=f"X-CSQA-{lang}",
            filter=lambda x: all(len(x["question"]["choices"]["text"][i].strip()) > 0 for i in range(len(x["question"]["choices"]["text"]))),
            evaluation_splits=("validation",),
            generation_size=-1,
            stop_sequence=("\n",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
                    Metrics.loglikelihood_acc_norm_token, Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi,
                    Metrics.prob_raw, Metrics.prob_raw_norm, Metrics.prob_raw_norm_token, Metrics.prob_raw_norm_pmi],
        )
