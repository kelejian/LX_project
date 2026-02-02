import logging
import logging.config
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

# 默认兜底日志配置
DEFAULT_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}

def setup_logger(
    save_dir: Optional[Union[str, Path]] = None,
    log_config: Optional[Union[str, Path]] = None,
    default_level: int = logging.INFO,
    name: str = None
) -> logging.Logger:
    """
    配置日志系统。
    
    Args:
        save_dir: 日志文件的保存目录（如果在 config json 中配置了文件输出）。
        log_config: 日志配置 json 文件的路径。
        default_level: 如果未找到配置文件，使用的默认日志等级。
        name: Logger 名称（可选）。
    
    Returns:
        logger: 配置好的 logger 实例。
    """
    if log_config:
        log_config = Path(log_config)
        if log_config.is_file():
            try:
                with open(log_config, 'rt') as f:
                    config = json.load(f)
                
                # 如果提供了 save_dir，更新文件处理器路径
                if save_dir:
                    save_dir = Path(save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    for _, handler in config.get('handlers', {}).items():
                        if 'filename' in handler:
                            # 将 save_dir 前缀添加到 filename
                            handler['filename'] = str(save_dir / Path(handler['filename']).name)
                
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"加载日志配置出错: {e}。使用默认配置。")
                logging.config.dictConfig(DEFAULT_LOG_CONFIG)
        else:
            print(f"警告: 未找到日志配置文件 {log_config}。使用默认配置。")
            logging.config.dictConfig(DEFAULT_LOG_CONFIG)
    else:
        logging.config.dictConfig(DEFAULT_LOG_CONFIG)
        
    logger = logging.getLogger(name)
    logger.setLevel(default_level)
    return logger
