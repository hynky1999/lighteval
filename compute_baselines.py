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

import os
import importlib
import importlib.util
from typing import Optional
from datasets import load_dataset

from typer import Argument, Option
from typing_extensions import Annotated
from accelerate import PartialState

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.logging.hierarchical_logger import hlog
from lighteval.metrics.utils import MetricCategory
from lighteval.models.model_loader import ModelInfo
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.registry import Registry, get_custom_tasks, taskinfo_selector
from lighteval.utils import as_list

# CACHE_DIR: str = os.getenv("HF_HOME", "")

HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Logging Parameters" 
HELP_PANEL_NAME_3 = "Debug Parameters"

def compute_baseline(
    tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
    cache_dir: Annotated[
        str, Option(help="Cache directory for datasets and models.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    custom_tasks: Annotated[
        Optional[str], Option(help="Path to custom tasks directory.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = None,
    dataset_loading_processes: Annotated[
        int, Option(help="Number of processes to use for dataset loading.", rich_help_panel=HELP_PANEL_NAME_1)
    ] = 1,
    output_dir: Annotated[
        str, Option(help="Output directory for evaluation results.", rich_help_panel=HELP_PANEL_NAME_2)
    ] = "results",
    max_samples: Annotated[
        Optional[int], Option(help="Maximum number of samples to evaluate on.", rich_help_panel=HELP_PANEL_NAME_3)
    ] = None,
    name: Annotated[
        str, Option(help="Name of the baseline.", rich_help_panel=HELP_PANEL_NAME_2)
    ] = "lighteval/0",
):
    """Compute baselines for given tasks."""
    # Initialize accelerate state
    _ = PartialState()

    # Get tasks using the registry approach
    if custom_tasks:
        _, tasks_groups_dict = get_custom_tasks(custom_tasks)
        selected_task_groups = tasks.split(",")
        if tasks_groups_dict and all(task in tasks_groups_dict for task in selected_task_groups):
            tasks = ",".join(tasks_groups_dict[task_group] for task_group in selected_task_groups)

    task_names_list, few_shots_dict = taskinfo_selector(tasks)
    task_dict = Registry(cache_dir=cache_dir).get_task_dict(
        task_names_list,
        custom_tasks=custom_tasks,
    )

    # Initialize evaluation tracker
    evaluation_tracker = EvaluationTracker(logging_dir=output_dir)
    
    # Log model info
    evaluation_tracker.general_config_logger.log_model_info(
        ModelInfo(
            model_name=name,
            model_sha=None,
            model_dtype=None,
            model_size=None,
        )
    )
    
    # Log task configs
    evaluation_tracker.task_config_logger.log(task_dict)
    # Process each task
    LightevalTask.load_datasets(task_dict.values(), dataset_loading_processes)
    for task_name, task in task_dict.items():
        task_docs = list(task.eval_docs())
        n_samples = min(max_samples, len(task_docs)) if max_samples else len(task_docs)

        p_correct_score = [
            len(as_list(task_doc.gold_index)) / len(task_doc.choices) for task_doc in task_docs[:n_samples]
        ]

        metric_results = {
            metric.metric: p_correct_score
            if metric.category
            in [MetricCategory.MULTICHOICE, MetricCategory.MULTICHOICE_PMI, MetricCategory.MULTICHOICE_ONE_TOKEN]
            else 0
            for metric in task.metrics
        }

        for fewshots, _ in few_shots_dict[task_name]:
            evaluation_tracker.metrics_logger.log(f"{task_name}|{fewshots}", metric_results)

    evaluation_tracker.metrics_logger.aggregate(task_dict=task_dict, bootstrap_iters=1000)
    evaluation_tracker.save(save_results=True, save_details=True, save_tensorboard=True)

if __name__ == "__main__":
    import typer
    typer.run(compute_baseline)