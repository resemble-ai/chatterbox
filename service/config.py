# CRITICAL = 50
# ERROR = 40
# WARNING = 30
# INFO = 20
# DEBUG = 10
# NOTSET = 0
__logger_cfg__ = dict(
    logger_name="root",
    file_level=10,
    console_level=20,
    logger_path='logs/server.log',
)

type = 'FastAPIServer'
checkpoint_dir = "weights"
audio_prompts_dir = "data"
enable_cors = True
host = '0.0.0.0'
port = 18085
logger_cfg = __logger_cfg__
