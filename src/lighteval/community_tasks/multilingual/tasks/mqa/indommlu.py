from ast import List
from typing import Literal

from ..utils.prompts import get_ar_mmlu_prompt, get_cmllu_prompt, get_indommlu_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


IMMLU_TASK_TYPE = Literal[
    'Sejarah', 
    'Geografi', 
    'Bahasa Lampung', 
    'IPS', 
    'Bahasa Bali', 
    'Bahasa Makassar', 
    'Bahasa Banjar', 
    'Kimia', 
    'Biologi', 
    'IPA', 
    'Agama Kristen', 
    'Kesenian', 
    'Agama Islam', 
    'Agama Hindu', 
    'Bahasa Madura', 
    'Penjaskes', 
    'Bahasa Indonesia', 
    'Fisika', 
    'Budaya Alam Minangkabau', 
    'Bahasa Dayak Ngaju', 
    'Sosiologi', 
    'Ekonomi', 
    'Bahasa Sunda', 
    'Bahasa Jawa', 
    'PPKN'
]

class IndonesianMMLUTask(LightevalTaskConfig):
    def __init__(self, task: IMMLU_TASK_TYPE):
        self.task = task
        super().__init__(
            name=f"mmlu-id:{task}",
            prompt_function=get_indommlu_prompt("id"),
            suite=("custom",),
            hf_repo="indolem/IndoMMLU",
            hf_subset="default",
            trust_dataset=True,
            evaluation_splits=("test",),
            filter=lambda line: line["subject"] == task,
            few_shots_split="train",
            metric=(
                Metrics.loglikelihood_acc,
                Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token,
                Metrics.loglikelihood_acc_norm_pmi,
                Metrics.brier_score_norm,
                Metrics.brier_score_token,
                Metrics.brier_score_pmi,
                Metrics.brier_score,
                Metrics.loglikelihood_prob,
                Metrics.loglikelihood_prob_norm,
                Metrics.loglikelihood_prob_norm_token,
                Metrics.loglikelihood_prob_norm_pmi, 
            ),
        )
