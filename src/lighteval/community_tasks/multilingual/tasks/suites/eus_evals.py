from typing import Literal
from lighteval.community_tasks.multilingual.tasks.utils.prompts import get_basque_reading_proficiency_prompt, get_basque_reading_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


class BasqueRC(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"rc-eu",
            prompt_function=get_basque_reading_prompt("eu"),
            suite=("custom",),
            hf_repo="HiTZ/EusReading",
            hf_subset="default",
            evaluation_splits=("test",),
            metric=(Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token),
        )

    
class BasqueTrivia(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"trivia-eu",
            prompt_function=get_basque_reading_prompt("eu"),
            suite=("custom",),
            hf_repo="HiTZ/EusTrivia",
            hf_subset="default",
            evaluation_splits=("test",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token
            ]
        )

EXAMS_SUBSETS = ['eu_opeosakiadmineu',
 'eu_opeosakiauxenfeu',
 'eu_opeosakiauxeu',
 'eu_opeosakiceladoreu',
 'eu_opeosakienfeu',
 'eu_opeosakioperarioeu',
 'eu_opeosakitecnicoeu',
 'eu_opeosakivarioseu',
 'eu_opegasteizkoudala',
 'eu_opeehuadmineu',
 'eu_opeehuauxeu',
 'eu_opeehubiblioeu',
 'eu_opeehuderechoeu',
 'eu_opeehueconomicaseu',
 'eu_opeehuempresarialeseu',
 'eu_opeehusubalternoeu',
 'eu_opeehutecnicoeu',
 'eu_opeehuteknikarib',
 'eu_ejadministrari',
 'eu_ejlaguntza',
 'eu_ejlaguntzaile',
 'eu_ejteknikari',
 'eu_osakidetza1e',
 'eu_osakidetza2e',
 'eu_osakidetza3e',
 'eu_osakidetza5e',
 'eu_osakidetza6e',
 'eu_osakidetza7e',
 'eu_opebilbaoeu',
#  'es_opeosakiadmin',
#  'es_opeosakiaux',
#  'es_opeosakiauxenf',
#  'es_opeosakicelador',
#  'es_opeosakienf',
#  'es_opeosakijuridico',
#  'es_opeosakioperario',
#  'es_opeosakitecnico',
#  'es_opeosakivarios',
#  'es_opeayuntamientovitoria',
#  'es_opeehuadmin',
#  'es_opeehuaux',
#  'es_opeehubiblio',
#  'es_opeehuderecho',
#  'es_opeehueconomicas',
#  'es_opeehuempresariales',
#  'es_opeehusubalterno',
#  'es_opeehutecnico',
#  'es_opeehutecnicob',
#  'es_ejadministrativo',
#  'es_ejauxiliar',
#  'es_ejsubalterno',
#  'es_ejtecnico',
#  'es_osakidetza1c',
#  'es_osakidetza2c',
#  'es_osakidetza3c',
#  'es_osakidetza4c',
#  'es_osakidetza5c',
#  'es_osakidetza6c',
#  'es_osakidetza7c',
#  'es_osakidetza8c',
#  'es_osakidetza9c',
#  'es_opebilbao']
]
# TODO: switch between spanish
class BasqueExams(LightevalTaskConfig):
    def __init__(self, subset: str):
        super().__init__(
            name=f"exams-eu:{subset}",
            prompt_function=get_basque_reading_prompt("eu"),
            suite=("custom",),
            hf_repo="HiTZ/EusExams",
            hf_subset=subset,
            filter=lambda line: line["answer"] is not None,
            evaluation_splits=("test",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token
            ]
        )
    

class BasqueReadingProficiency(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name=f"reading-proficiency-eu",
            prompt_function=get_basque_reading_proficiency_prompt("eu"),
            suite=("custom",),
            hf_repo="HiTZ/EusProficiency",
            hf_subset="default",
            evaluation_splits=("test",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token
            ]
        )


class BertaQATask(LightevalTaskConfig):
    def __init__(self, lang: Literal["eu"]):
        super().__init__(
            name=f"bartaqa-{lang}",
            prompt_function=get_basque_reading_prompt(lang),
            suite=("custom",),
            hf_repo="HiTZ/BertaQA",
            hf_subset="eu",
            evaluation_splits=("test",),
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace,
                Metrics.loglikelihood_acc_norm_token, Metrics.loglikelihood_prob, Metrics.loglikelihood_prob_norm, Metrics.loglikelihood_prob_norm_token
            ]
        )