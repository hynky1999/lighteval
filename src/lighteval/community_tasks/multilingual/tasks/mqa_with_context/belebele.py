# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Literal

from datasets import get_dataset_split_names
from langcodes import standardize_tag

from ..utils.prompts import get_m_belebele_prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


# TODO all supported langauges
LANGS = Literal["ar", "en", "bg", "hr", "hu", "it", "mk", "pl", "pt", "sq", "sr", "tr", "vi", "zh", "te", "th", "sw", "hi", "ru", "fr", "eu"]


# We convert from made-up codes to "standard" codes 😎, ignoring the script
class BelebeleTask(LightevalTaskConfig):
    def __init__(self, lang: LANGS):
        # TODO: beleble now uses actually config name, until i update the code I am locking the revision
        if lang == "zh":
            splits = ["zho_Hans"]
        else:
            splits = [
                split
                for split in get_dataset_split_names("facebook/belebele", revision="75b399394a9803252cfec289d103de462763db7c")
                if standardize_tag(split, macro=True) == lang
            ]
        if len(splits) != 1:
            raise ValueError(
                f"Language {lang} not found in belebele or there are multiple splits for the same language"
            )
        split = splits[0]
        super().__init__(
            name=f"belebele-{lang}",
            prompt_function=get_m_belebele_prompt(lang),
            suite=("custom",),
            hf_revision="75b399394a9803252cfec289d103de462763db7c",
            hf_repo="facebook/belebele",
            hf_subset="default",
            evaluation_splits=(split,),
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
