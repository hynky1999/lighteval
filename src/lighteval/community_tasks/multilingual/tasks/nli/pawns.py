from typing import Literal

from lighteval.community_tasks.multilingual.tasks.utils.translation_literals import FULL_STOP

from ..utils.prompts import get_paws_x_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


LANGS = Literal["de", "en", "es", "fr", "ja", "ko", "zh"]


class PawnsXTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS, version: Literal[1, 2]):
        super().__init__(
            name=f"pawns{f'-v{version}' if version != 1 else ''}-{lang}",
            suite=("custom",),
            prompt_function=get_paws_x_prompt(lang, version=version),
            hf_repo="google-research-datasets/paws-x",
            hf_subset=lang,
            filter=lambda x: x["sentence1"].endswith(FULL_STOP[lang]),
            evaluation_splits=("test",),
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
