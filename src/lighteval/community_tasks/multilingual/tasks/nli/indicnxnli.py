from typing import Literal


from ..utils.translation_literals import FULL_STOP
from ..utils.prompts import get_xnli_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]


class XNLIIndicTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS, version: Literal[1,2]):
        super().__init__(
            name=f"indicnxnli-{lang}-bool{f'-v{version}' if version != 1 else ''}-{lang}",
            suite=("custom",),
            prompt_function=get_xnli_prompt(lang, version),
            hf_repo="Divyanshu/indicxnli",
            hf_subset=lang,
            # Ignore neutral
            filter=lambda x: x["premise"].endswith(FULL_STOP[lang]) and int(x["label"]) in [0, 2],
            evaluation_splits=("validation",),
            few_shots_split="train",
            few_shots_select=None,
            generation_size=-1,
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                                Metrics.thresholded_prob_norm,
                Metrics.thresholded_prob_token,
                Metrics.thresholded_prob_pmi,
                Metrics.thresholded_prob,
                Metrics.brier_score_norm,
                Metrics.brier_score_token,
                Metrics.brier_score_pmi,
                Metrics.brier_score,
                Metrics.one_minus_distance_to_dominant_prob_norm,
                Metrics.one_minus_distance_to_dominant_prob_norm_token,
                Metrics.one_minus_distance_to_dominant_prob_norm_pmi,
                Metrics.one_minus_distance_to_dominant_prob,
                Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi, Metrics.prob_raw,  Metrics.prob_raw_norm, Metrics.prob_raw_norm_token,  Metrics.prob_raw_norm_pmi, 
            ),
        )
