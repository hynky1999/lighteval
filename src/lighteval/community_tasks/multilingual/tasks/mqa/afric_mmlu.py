
from typing import Literal
from lighteval.community_tasks.multilingual.tasks.mqa.mlmm import MMLU_SUBSET
from lighteval.community_tasks.multilingual.tasks.utils.prompts import get_mmlu_prompt, get_cmllu_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

class AfricMMLUTask(LightevalTaskConfig):
    def __init__(self, subset: MMLU_SUBSET):
        super().__init__(
            name=f"afric-mmlu-sw:{subset}",
            prompt_function=get_mmlu_prompt("sw"),
            suite=("custom",),
            hf_repo="masakhane/afrimmlu",
            hf_revision="refs/pr/1",
            hf_subset="swa",
            filter=lambda line: line["subject"] == subset,
            trust_dataset=True,
            evaluation_splits=("test",),
            few_shots_split="dev",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_token,
                Metrics.loglikelihood_prob_norm_pmi, Metrics.prob_raw,  Metrics.prob_raw_norm, Metrics.prob_raw_norm_token,  Metrics.prob_raw_norm_pmi, 
            ),
        )
        self.subset = subset

class OpenAIMMLUTask(LightevalTaskConfig):
    def __init__(self, subset: MMLU_SUBSET):
        super().__init__(
            name=f"openai-mmlu-sw:{subset}",
            prompt_function=get_cmllu_prompt("sw"),
            suite=("custom",),
            hf_repo="openai/MMMLU",
            hf_subset="SW_KE",
            filter=lambda x: x["Subject"].lower() == subset,
            trust_dataset=True,
            evaluation_splits=("test",),
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_token,
                Metrics.loglikelihood_prob_norm_pmi, Metrics.prob_raw,  Metrics.prob_raw_norm, Metrics.prob_raw_norm_token,  Metrics.prob_raw_norm_pmi, 
            ),
        )
        self.subset = subset