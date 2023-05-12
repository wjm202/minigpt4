"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import json
from typing import Dict
from omegaconf import OmegaConf
from minigpt4.common.registry import registry


class Config:

    def __init__(self, args):
        self.config = {}
        self.args = args
        registry.register('configuration', self)
        user_config = self._build_opt_list(self.args.options)
        config = OmegaConf.load(self.args.cfg_path)
        runner_config = self.build_runner_config(config)
        model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(config)
        self.config = OmegaConf.merge(runner_config, model_config,
            dataset_config, user_config)

    def _validate_runner_config(self, runner_config):
        """
        This method validates the configuration, such that
            1) all the user specified options are valid;
            2) no type mismatches between the user specified options and the config.
        """
        runner_config_validator = create_runner_config_validator()
        runner_config_validator.validate(runner_config)

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    @staticmethod
    def build_model_config(config, **kwargs):
        model = config.get('model', None)
        assert model is not None, 'Missing model configuration file.'
        model_cls = registry.get_model_class(model.arch)
        assert model_cls is not None, f"Model '{model.arch}' has not been registered."
        model_type = kwargs.get('model.model_type', None)
        if not model_type:
            model_type = model.get('model_type', None)
        assert model_type is not None, 'Missing model_type.'
        model_config_path = model_cls.default_config_path(model_type=model_type
            )
        model_config = OmegaConf.create()
        model_config = OmegaConf.merge(model_config, OmegaConf.load(
            model_config_path), {'model': config['model']})
        return model_config

    @staticmethod
    def build_runner_config(config):
        return {'run': config.run}

    @staticmethod
    def build_dataset_config(config):
        datasets = config.get('datasets', None)
        if datasets is None:
            raise KeyError(
                "Expecting 'datasets' as the root key for dataset configuration."
                )
        dataset_config = OmegaConf.create()
        for dataset_name in datasets:
            builder_cls = registry.get_builder_class(dataset_name)
            dataset_config_type = datasets[dataset_name].get('type', 'default')
            dataset_config_path = builder_cls.default_config_path(type=
                dataset_config_type)
            dataset_config = OmegaConf.merge(dataset_config, OmegaConf.load
                (dataset_config_path), {'datasets': {dataset_name: config[
                'datasets'][dataset_name]}})
        return dataset_config

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []
        if len(opts) == 0:
            return opts
        has_equal = opts[0].find('=') != -1
        if has_equal:
            return opts
        return [(opt + '=' + value) for opt, value in zip(opts[0::2], opts[
            1::2])]

    def get_config(self):
        return self.config

    @property
    def run_cfg(self):
        return self.config.run

    @property
    def datasets_cfg(self):
        return self.config.datasets

    @property
    def model_cfg(self):
        return self.config.model

    def pretty_print(self):
        logging.info('\n=====  Running Parameters    =====')
        logging.info(self._convert_node_to_json(self.config.run))
        logging.info('\n======  Dataset Attributes  ======')
        datasets = self.config.datasets
        for dataset in datasets:
            if dataset in self.config.datasets:
                logging.info(f'\n======== {dataset} =======')
                dataset_config = self.config.datasets[dataset]
                logging.info(self._convert_node_to_json(dataset_config))
            else:
                logging.warning(
                    f"No dataset named '{dataset}' in config. Skipping")
        logging.info(f'\n======  Model Attributes  ======')
        logging.info(self._convert_node_to_json(self.config.model))

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)


def node_to_dict(node):
    return OmegaConf.to_container(node)


class ConfigValidator:
    """
    This is a preliminary implementation to centralize and validate the configuration.
    May be altered in the future.

    A helper class to validate configurations from yaml file.

    This serves the following purposes:
        1. Ensure all the options in the yaml are defined, raise error if not.
        2. when type mismatches are found, the validator will raise an error.
        3. a central place to store and display helpful messages for supported configurations.

    """


    class _Argument:

        def __init__(self, name, choices=None, type=None, help=None):
            self.name = name
            self.val = None
            self.choices = choices
            self.type = type
            self.help = help

        def __str__(self):
            s = f'{self.name}={self.val}'
            if self.type is not None:
                s += f', ({self.type})'
            if self.choices is not None:
                s += f', choices: {self.choices}'
            if self.help is not None:
                s += f', ({self.help})'
            return s

    def __init__(self, description):
        self.description = description
        self.arguments = dict()
        self.parsed_args = None

    def __getitem__(self, key):
        assert self.parsed_args is not None, 'No arguments parsed yet.'
        return self.parsed_args[key]

    def __str__(self) ->str:
        return self.format_help()

    def add_argument(self, *args, **kwargs):
        """
        Assume the first argument is the name of the argument.
        """
        self.arguments[args[0]] = self._Argument(*args, **kwargs)

    def validate(self, config=None):
        """
        Convert yaml config (dict-like) to list, required by argparse.
        """
        for k, v in config.items():
            assert k in self.arguments, f'{k} is not a valid argument. Support arguments are {self.format_arguments()}.'
            if self.arguments[k].type is not None:
                try:
                    self.arguments[k].val = self.arguments[k].type(v)
                except ValueError:
                    raise ValueError(
                        f'{k} is not a valid {self.arguments[k].type}.')
            if self.arguments[k].choices is not None:
                assert v in self.arguments[k
                    ].choices, f'{k} must be one of {self.arguments[k].choices}.'
        return config

    def format_arguments(self):
        return str([f'{k}' for k in sorted(self.arguments.keys())])

    def format_help(self):
        help_msg = str(self.description)
        return help_msg + ', available arguments: ' + self.format_arguments()

    def print_help(self):
        print(self.format_help())


def create_runner_config_validator():
    validator = ConfigValidator(description='Runner configurations')
    validator.add_argument('runner', type=str, choices=['runner_base',
        'runner_iter'], help=
        """Runner to use. The "runner_base" uses epoch-based training while iter-based
            runner runs based on iters. Default: runner_base"""
        )
    validator.add_argument('train_dataset_ratios', type=Dict[str, float],
        help=
        """Ratios of training dataset. This is used in iteration-based runner.
        Do not support for epoch-based runner because how to define an epoch becomes tricky.
        Default: None"""
        )
    validator.add_argument('max_iters', type=float, help=
        'Maximum number of iterations to run.')
    validator.add_argument('max_epoch', type=int, help=
        'Maximum number of epochs to run.')
    validator.add_argument('iters_per_inner_epoch', type=float, help=
        'Number of iterations per inner epoch. This is required when runner is runner_iter.'
        )
    lr_scheds_choices = registry.list_lr_schedulers()
    validator.add_argument('lr_sched', type=str, choices=lr_scheds_choices,
        help='Learning rate scheduler to use, from {}'.format(
        lr_scheds_choices))
    task_choices = registry.list_tasks()
    validator.add_argument('task', type=str, choices=task_choices, help=
        'Task to use, from {}'.format(task_choices))
    validator.add_argument('init_lr', type=float, help=
        'Initial learning rate. This will be the learning rate after warmup and before decay.'
        )
    validator.add_argument('min_lr', type=float, help=
        'Minimum learning rate (after decay).')
    validator.add_argument('warmup_lr', type=float, help=
        'Starting learning rate for warmup.')
    validator.add_argument('lr_decay_rate', type=float, help=
        'Learning rate decay rate. Required if using a decaying learning rate scheduler.'
        )
    validator.add_argument('weight_decay', type=float, help=
        'Weight decay rate.')
    validator.add_argument('batch_size_train', type=int, help=
        'Training batch size.')
    validator.add_argument('batch_size_eval', type=int, help=
        'Evaluation batch size, including validation and testing.')
    validator.add_argument('num_workers', help=
        'Number of workers for data loading.')
    validator.add_argument('warmup_steps', type=int, help=
        'Number of warmup steps. Required if a warmup schedule is used.')
    validator.add_argument('seed', type=int, help='Random seed.')
    validator.add_argument('output_dir', type=str, help=
        'Output directory to save checkpoints and logs.')
    validator.add_argument('evaluate', help=
        'Whether to only evaluate the model. If true, training will not be performed.'
        )
    validator.add_argument('train_splits', type=list, help=
        'Splits to use for training.')
    validator.add_argument('valid_splits', type=list, help=
        'Splits to use for validation. If not provided, will skip the validation.'
        )
    validator.add_argument('test_splits', type=list, help=
        'Splits to use for testing. If not provided, will skip the testing.')
    validator.add_argument('accum_grad_iters', type=int, help=
        'Number of iterations to accumulate gradient for.')
    validator.add_argument('device', type=str, choices=['cpu', 'gpu'],
        help="Device to use. Support  'gpu' or 'cpu' as for now.")
    validator.add_argument('world_size', type=int, help=
        'Number of processes participating in the job.')
    validator.add_argument('dist_url', type=str)
    validator.add_argument('distributed', type=bool)
    validator.add_argument('use_dist_eval_sampler', type=bool, help=
        'Whether to use distributed sampler during evaluation or not.')
    validator.add_argument('max_len', type=int, help=
        'Maximal length of text output.')
    validator.add_argument('min_len', type=int, help=
        'Minimal length of text output.')
    validator.add_argument('num_beams', type=int, help=
        'Number of beams used for beam search.')
    validator.add_argument('num_ans_candidates', type=int, help=
        'For ALBEF and BLIP, these models first rank answers according to likelihood to select answer candidates.'
        )
    validator.add_argument('inference_method', type=str, choices=[
        'genearte', 'rank'], help=
        'Inference method to use for question answering. If rank, requires a answer list.'
        )
    validator.add_argument('k_test', type=int, help=
        'Number of top k most similar samples from ITC/VTC selection to be tested.'
        )
    return validator
