from typing import Literal

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

from ..utils.prompts import get_copa_prompt


LANGS = Literal["ar", "et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"]


class XCopaTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        repo = "xcopa" if lang != "ar" else "OALL/AlGhafa-Arabic-LLM-Benchmark-Translated"
        subset = lang if lang != "ar" else "copa_ext_ar"
        #TODO: The ar also has fewshots
        super().__init__(
            name=f"xcopa-{lang}",
            suite=("custom",),
            prompt_function=get_copa_prompt(lang),
            hf_repo=repo,
            hf_subset=subset,
            evaluation_splits=("test",),
            few_shots_split=None,
            few_shots_select=None,
            generation_size=-1,
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
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
                Metrics.one_minus_distance_to_dominant_prob, Metrics.loglikelihood_acc_norm_pmi, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token, Metrics.loglikelihood_prob_norm_pmi),
            stop_sequence=("\n",),
            trust_dataset=True,
            version=0,
        )

class XCopaTaskEU(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"xcopa-eu",
            suite=("custom",),
            prompt_function=get_copa_prompt("eu"),
            hf_repo="HiTZ/XCOPA-eu",
            hf_subset="xcopa",
            evaluation_splits=("test",),
            few_shots_split="validation",
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
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
                Metrics.one_minus_distance_to_dominant_prob, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token),
        )
