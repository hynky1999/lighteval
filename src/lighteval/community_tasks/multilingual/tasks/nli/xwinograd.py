from typing import Literal
from ..utils.prompts import get_winogrande_prompt

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["zh", "ru", "fr", "en"]


class XWinogradeTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        super().__init__(
            name=f"xwinograd-{lang}",
            suite=("custom",),
            prompt_function=get_winogrande_prompt(lang),
            hf_repo="Muennighoff/xwinograd",
            hf_subset=lang,
            evaluation_splits=("test",),
            few_shots_split=None,
            few_shots_select=None,
            generation_size=-1,
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,                 Metrics.thresholded_prob_norm,
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
                Metrics.one_minus_distance_to_dominant_prob, Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi),
            stop_sequence=("\n",),
            output_regex=None,
            version=0,
        )


#TODO: Add thai wsc (it's a bit non-trivial and time consuming)
# https://huggingface.co/datasets/pakphum/winograd_th?row=2
# https://github.com/PhakphumAdev/Thai-Winograd/blob/main/prompt_eval.py
# https://huggingface.co/datasets/pakphum/winograd_th?row=2