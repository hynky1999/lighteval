from ast import List
from typing import Literal

from ..utils.prompts import get_ar_mmlu_prompt, get_cmllu_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


JMMLU_TASK_TYPE = Literal[
    'japanese_history', 
    'miscellaneous', 
    'security_studies', 
    'virology', 
    'nutrition', 
    'human_sexuality', 
    'college_mathematics', 
    'japanese_civics', 
    'econometrics', 
    'computer_security', 
    'clinical_knowledge', 
    'machine_learning', 
    'high_school_chemistry', 
    'human_aging', 
    'logical_fallacies', 
    'sociology', 
    'high_school_european_history', 
    'high_school_statistics', 
    'high_school_physics', 
    'high_school_microeconomics', 
    'college_physics', 
    'anatomy', 
    'high_school_psychology', 
    'business_ethics', 
    'professional_psychology', 
    'college_medicine', 
    'elementary_mathematics', 
    'moral_disputes', 
    'marketing', 
    'high_school_macroeconomics', 
    'world_religions',
    'conceptual_physics',
    'professional_medicine',
    'prehistory',
    'high_school_mathematics',
    'international_law',
    'philosophy',
    'japanese_idiom',
    'japanese_geography',
    'management',
    'high_school_computer_science',
    'medical_genetics',
    'college_computer_science',
    'public_relations',
    'professional_accounting',
    'abstract_algebra',
    'global_facts',
    'college_biology',
    'high_school_geography',
    'world_history',
    'high_school_biology',
    'college_chemistry',
    'electrical_engineering',
    'astronomy',
    'jurisprudence',
    'formal_logic'
]

class JapaneseMMLUTask(LightevalTaskConfig):
    def __init__(self, task: JMMLU_TASK_TYPE):
        self.task = task
        super().__init__(
            name=f"mmlu-jp:{task}",
            prompt_function=get_ar_mmlu_prompt("jp"),
            suite=("custom",),
            hf_repo="nlp-waseda/JMMLU",
            trust_dataset=True,
            hf_subset=task,
            evaluation_splits=("test",),
            few_shots_split="test",
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
