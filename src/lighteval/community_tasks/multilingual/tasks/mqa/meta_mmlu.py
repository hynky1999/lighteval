from typing import Literal

from lighteval.community_tasks.multilingual.tasks.utils.prompts import get_meta_mmlu_prompt
from .mlmm import MMLU_SUBSET
from lighteval.metrics.metrics import Metrics 

from lighteval.tasks.lighteval_task import LightevalTaskConfig
LANGS = Literal["de", "es", "fr", "hi", "it", "pt", "th"]



class MetaMMLUTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS, task: MMLU_SUBSET):
        super().__init__(
            name=f"meta_mmlu-{lang}:{task}",
            prompt_function=get_meta_mmlu_prompt(lang),
            suite=("custom",),
            hf_repo="meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
            hf_subset=f"Llama-3.1-8B-Instruct-evals__multilingual_mmlu_{lang}__details",
            filter=lambda line: line["subtask_name"] == f"mmlu_{lang}_chat.{task}",
            evaluation_splits=("latest",),
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