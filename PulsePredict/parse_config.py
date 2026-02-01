import os, sys
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None, is_test_run=False, is_finetune=False):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        :param is_test_run: Flag to indicate if the run is for testing.
        :param is_finetune: Flag to indicate if the run is for fine-tuning.
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
        save_dir_root = Path(self.config['trainer']['save_dir'])
        exper_name = self.config['name']

        save_dir_root = save_dir_root.resolve()

        # 根据运行模式决定并同步创建 models 和 log 目录
        if self.resume and not is_finetune:
            # --- 恢复训练 (RESUME) 或 测试 (TEST) 模式 ---
            # 确定父级目录
            parent_model_dir = Path(self.resume).parent

            # 1. 定义 models 和 log 的根目录 (现在将是绝对路径)
            models_root = save_dir_root / 'models'
            log_root = save_dir_root / 'log'
            # 2. 获取 model 目录相对于其根目录的路径
            relative_model_path = parent_model_dir.relative_to(models_root)
            # 3. 在 log 根目录下重建这个相对路径，得到镜像的父级日志目录
            parent_log_dir = log_root / relative_model_path
            
            session_name = f"test_{timestamp}" if is_test_run else f"resume_{timestamp}"
            
            self._save_dir = parent_model_dir / session_name
            self._log_dir = parent_log_dir / session_name
            
            session_name = f"test_{timestamp}" if is_test_run else f"resume_{timestamp}"
            
            self._save_dir = parent_model_dir / session_name
            self._log_dir = parent_log_dir / session_name

            # 创建新的会话子目录
            self.save_dir.mkdir(parents=True, exist_ok=False)
            self.log_dir.mkdir(parents=True, exist_ok=False)
            
            # 将当前配置的副本写入新的模型会话子目录
            write_json(self.config, self.save_dir / 'config.json')

        else:
            # --- 从零训练 (NEW TRAINING) 或 微调 (FINETUNING) 模式 ---
            if is_finetune:
                exper_name = f"{exper_name}_finetuned"
            
            run_id = timestamp if run_id is None else run_id
            self._save_dir = save_dir_root / 'models' / exper_name / run_id
            self._log_dir = save_dir_root / 'log' / exper_name / run_id

            # 创建全新的顶级目录
            self.save_dir.mkdir(parents=True, exist_ok=False)
            self.log_dir.mkdir(parents=True, exist_ok=False)

            # 将配置写入新的顶级模型目录
            write_json(self.config, self.save_dir / 'config.json')

        # 统一的日志模块配置，指向最终确定的log_dir
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        
        # 根据调用脚本和参数判断运行模式
        is_test_run = 'test.py' in sys.argv[0]
        
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        
        is_finetune = False
        if args.config and resume:
            # 如果同时提供了-c和-r，则认为是微调模式，并用新config更新
            config.update(read_json(args.config))
            is_finetune = True

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        
        return cls(config, resume, modification, is_test_run=is_test_run, is_finetune=is_finetune)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
