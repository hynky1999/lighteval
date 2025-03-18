
from typing import Literal
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from ..utils.prompts import get_wsci_prompt


class WSCITask(LightevalTaskConfig):
    def __init__(self, lang: Literal["th"]):
        self.lang = lang
        super().__init__(
            name=f"wsci-{lang}",
            prompt_function=get_wsci_prompt(lang),
            suite=("custom",),
            hf_repo="pakphum/winograd_th",
            hf_subset=f"default",
            evaluation_splits=("test",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
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
                Metrics.one_minus_distance_to_dominant_prob, Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi],
        )
