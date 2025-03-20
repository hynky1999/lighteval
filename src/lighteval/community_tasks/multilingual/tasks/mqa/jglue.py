from typing import Literal
from ..utils.metrics import get_qa_metric
from ..utils.prompts import get_c3_prompt, get_cmnli_prompt, get_commonsenseqa_prompt, get_jnli_prompt, get_mlqa_prompt, get_ocnli_prompt, get_xnli_prompt
from lighteval.metrics.metrics import Metrics
from ..utils.translation_literals import FULL_STOP
from lighteval.tasks.lighteval_task import LightevalTaskConfig

class JampTask(LightevalTaskConfig):
    def __init__(self, version: Literal[1,2]):
        super().__init__(
            name=f"jamp-bool{f'-v{version}' if version != 1 else ''}-jp",
            prompt_function=get_xnli_prompt("jp", version),
            suite=("custom",),
            hf_repo="zenless-lab/jamp",
            hf_subset="default",
            # Only keep the positive and negative examples
            filter=lambda x: int(x["label"]) in [1, 2],
            evaluation_splits=("test",),
            few_shots_split="train",
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

class JNLI(LightevalTaskConfig):
    def __init__(self, version: Literal[1,2]):
        super().__init__(
            name=f"jnli-bool{f'-v{version}' if version != 1 else ''}-jp",
            prompt_function=get_jnli_prompt("jp", version),
            suite=("custom",),
            hf_repo="shunk031/JGLUE",
            hf_subset="JNLI",
            trust_dataset=True,
            # Only keep the positive and negative examples
            filter=lambda x: int(x["label"]) in [0, 1],
            evaluation_splits=("test",),
            few_shots_split="train",
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

class JCommonsenseQA(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"commonsenseqa-jp",
            prompt_function=get_commonsenseqa_prompt("jp"),
            suite=("custom",),
            hf_repo="leemeng/jcommonsenseqa-v1.1",
            hf_subset="default",
            trust_dataset=True,
            evaluation_splits=("validation",),
            few_shots_split="train",
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
    