
# from typing import Literal

# from langcodes import standardize_tag
# from lighteval.community_tasks.multilingual.tasks.mqa.mlmm import MMLU_SUBSET
# from lighteval.community_tasks.multilingual.tasks.utils.prompts import get_mgsm_prompt, get_mmlu_prompt, get_cmllu_prompt
# from lighteval.comm
# from lighteval.metrics.metrics import Metrics
# from lighteval.tasks.lighteval_task import LightevalTaskConfig

# class AfricMMLUTask(LightevalTaskConfig):
#     def __init__(self, subset: MMLU_SUBSET):
#         super().__init__(
#             name=f"afric-mmlu-sw:{subset}",
#             prompt_function=get_mmlu_prompt("sw"),
#             suite=("custom",),
#             hf_repo="masakhane/afrimmlu",
#             hf_revision="refs/pr/1",
#             hf_subset="swa",
#             filter=lambda line: line["subject"] == subset,
#             trust_dataset=True,
#             evaluation_splits=("test",),
#             few_shots_split="dev",
#             metric=(
#                 Metrics.loglikelihood_acc,
#                 Metrics.loglikelihood_acc_norm_nospace,
#                                 Metrics.thresholded_prob_norm,
                Metrics.thresholded_prob_token,
                Metrics.thresholded_prob_pmi,
                Metrics.thresholded_prob,
                Metrics.brier_score_norm,
                Metrics.brier_score_token,
                Metrics.brier_score_pmi
                Metrics.brier_score,
                Metrics.one_minus_distance_to_dominant_prob_norm,
                Metrics.one_minus_distance_to_dominant_prob_norm_token,
                Metrics.one_minus_distance_to_dominant_prob_norm_pmi,
                Metrics.one_minus_distance_to_dominant_prob,
#                 Metrics.loglikelihood_acc_norm_pmi,
#                 Metrics.loglikelihood_prob,
#                 Metrics.loglikelihood_prob_norm,
#                 Metrics.loglikelihood_prob_norm_token,
#                 Metrics.loglikelihood_prob_norm_pmi, Metrics.prob_raw,  Metrics.prob_raw_norm, Metrics.prob_raw_norm_token,  Metrics.prob_raw_norm_pmi, 
#             ),
#         )
#         self.subset = subset

# subset_lang = {
#     "sw": "SW_KE",
#     "ar": "AR_XY",
#     "fr": "FR_FR",
#     "hi": "HI_IN",
#     "zh": "ZH_CN",

# }
# class OpenAIMMLUTask(LightevalTaskConfig):
#     def __init__(self, subset: MMLU_SUBSET, lang: Literal["sw", "ar", "fr", "hi", "zh"]):
#         super().__init__(
#             name=f"openai-mmlu-{lang}:{subset}",
#             prompt_function=get_cmllu_prompt(lang),
#             suite=("custom",),
#             hf_repo="openai/MMMLU",
#             hf_subset=subset_lang[lang],
#             hf_revision="038c7808122969ead7456361af05cb8f47d247f8",
#             filter=lambda x: x["Subject"].lower() == subset,
#             trust_dataset=True,
#             evaluation_splits=("test",),
#             metric=(
#                 Metrics.loglikelihood_acc,
#                 Metrics.loglikelihood_acc_norm_nospace,
#                                 Metrics.thresholded_prob_norm,
                Metrics.thresholded_prob_token,
                Metrics.thresholded_prob_pmi,
                Metrics.thresholded_prob,
                Metrics.brier_score_norm,
                Metrics.brier_score_token,
                Metrics.brier_score_pmi
                Metrics.brier_score,
                Metrics.one_minus_distance_to_dominant_prob_norm,
                Metrics.one_minus_distance_to_dominant_prob_norm_token,
                Metrics.one_minus_distance_to_dominant_prob_norm_pmi,
                Metrics.one_minus_distance_to_dominant_prob,
#                 Metrics.loglikelihood_acc_norm_pmi,
#                 Metrics.loglikelihood_prob,
#                 Metrics.loglikelihood_prob_norm,
#                 Metrics.loglikelihood_prob_norm_token,
#                 Metrics.loglikelihood_prob_norm_pmi, Metrics.prob_raw,  Metrics.prob_raw_norm, Metrics.prob_raw_norm_token,  Metrics.prob_raw_norm_pmi, 
#             ),
#         )
#         self.subset = subset

# class MGSMTask(LightevalTaskConfig):
#     def __init__(self, lang: Literal["sw", "fr", "ru", "zh", "th", "te"]):
#         super().__init__(
#             name=f"mgsm-{lang}",
#             prompt_function=get_mgsm_prompt(lang),
#             suite=("lighteval",),
#             hf_repo="juletxara/mgsm",
#             hf_subset=lang,
#             evaluation_splits=("test",),
#             few_shots_split="train",
#             generation_size=25,
#             metric=[
#                 multilingual_quasi_exact_match_metric(lang, "full"),
#             ],
#             stop_sequence=("\n",),
#         )

# class AfricMGSMTask(LightevalTaskConfig):
#     def __init__(self, lang: Literal["sw"]):
#         super().__init__(
#             name=f"afric-mgsm-{lang}",
#             prompt_function=get_mgsm_prompt(lang),
#             suite=("lighteval",),
#             hf_repo="masakhane/mgsm",
#             hf_subset=lang,
#             evaluation_splits=("test",),
#             few_shots_split="train",
#             metric=[
#                 multilingual_quasi_exact_match_metric(lang, "full"),
#             ],
#             generation_size=25,
#         )