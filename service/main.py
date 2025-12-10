import argparse
import os
import sys

from service.server import FastAPIServer
from service.utils import file2dict


def main(args) -> int:
    """Initialize and run the FastAPI server.

    Creates the logs directory if it doesn't exist, loads configuration
    from the specified file, initializes the appropriate server instance,
    and starts the server.

    Args:
        args (argparse.Namespace):
            Parsed command line arguments containing configuration file
            path and other application settings.

    Returns:
        int:
            Exit code. Returns 0 on successful execution.
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')
    startup_config = file2dict(args.config_path)
    logger_cfg = startup_config['logger_cfg'].copy()
    logger_cfg['logger_name'] = 'main'
    # init server
    cls_name = startup_config.pop('type')
    if cls_name == 'FastAPIServer':
        server = FastAPIServer(**startup_config)
    else:
        raise ValueError(f'Invalid server type: {cls_name}')
    server.run()
    return 0

def setup_parser():
    """Set up command line argument parser for the application.

    This function creates and configures an argument parser to handle
    command line arguments for the application, including configuration
    file path specification.

    Returns:
        argparse.Namespace:
            Parsed command line arguments containing configuration
            file path and other application settings.
    """
    parser = argparse.ArgumentParser(
        description='FastAPI server for text-to-speech audio generation',
        epilog='Example: python main.py --config_path service/online_config.py'
    )
    # server args
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to the configuration file containing server settings',
        default='service/config.py')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    ret_val = main(args)
    sys.exit(ret_val)
