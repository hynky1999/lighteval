from ..utils.metrics import get_qa_metric
from ..utils.prompts import (
    get_arc_prompt,
    get_french_boolqa_prompt,
    get_french_trivia_prompt,
    get_hellaswag_prompt,
    get_mlqa_prompt,
)
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


class FrenchARCTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="french-arc",
            prompt_function=get_arc_prompt("fr"),
            suite=("custom",),
            hf_repo="manu/french_bench_arc_challenge",
            hf_subset="default",
            evaluation_splits=("test",),
            few_shots_split="train",
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

class FrenchHellaSwagTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="french-hellaswag",
            prompt_function=get_hellaswag_prompt("fr", use_activity_label=False),
            suite=("custom",),
            hf_repo="manu/french_bench_hellaswag",
            hf_subset="default",
            evaluation_splits=("validation",),
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

# Possible use "D'après l'information dans le contexte donné, quelle est la réponse à la question ?"
class BoolQAFrenchTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="french-boolqa",
            prompt_function=get_french_boolqa_prompt("fr"),
            suite=("custom",),
            hf_repo="manu/french_boolq",
            hf_subset="default",
            evaluation_splits=("test",),
            few_shots_split="valid",
            generation_size=1,
            stop_sequence=["\n"],
            metric=(
                get_qa_metric("fr", "exact"),
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


class FQuADv2Task(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="fquadv2",
            prompt_function=get_mlqa_prompt("fr"),
            suite=("custom",),
            hf_repo="manu/fquad2_test",
            hf_subset="default",
            evaluation_splits=("test_hasAns",),
            few_shots_split="valid_hasAns",
            generation_size=80,
            stop_sequence=("\n",),
            metric=(get_qa_metric("fr", "exact"), get_qa_metric("fr", "f1"), get_qa_metric("fr", "recall"), get_qa_metric("fr", "contains")),
            # metric=(Metrics.prob_raw, Metrics.prob_raw_norm, Metrics.prob_raw_norm_token),
        )
    

class TriviaFrenchTask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="french-triviaqa",
            prompt_function=get_french_trivia_prompt("fr"),
            suite=("custom",),
            hf_repo="manu/french-trivia",
            hf_subset="default",
            evaluation_splits=("train",),
            generation_size=100,
            stop_sequence=("\n",),
            metric=(get_qa_metric("fr", "exact"), get_qa_metric("fr", "f1"), get_qa_metric("fr", "recall"), get_qa_metric("fr", "contains")),
        )


# FrenchBench Fquad multi is a bit strange imo

_GENERATIVE_TASKS = [
    FQuADv2Task(),
    TriviaFrenchTask(),
    BoolQAFrenchTask(),
]

_MC_TASKS = [
    FrenchARCTask(),
    FrenchHellaSwagTask(),
    BoolQAFrenchTask(),
]