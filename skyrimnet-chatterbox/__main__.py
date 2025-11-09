#!/usr/bin/env python3
"""
Entry point for running skyrimnet-chatterbox as a module:
    python -m skyrimnet-chatterbox
"""
from loguru import logger

if __name__ == "__main__":
    try:
        # Try relative import first (when run as module: python -m skyrimnet-chatterbox)
        from .skyrimnet_chatterbox import (
            parse_arguments,
            set_seed,
            load_model,
            demo,
            clear_output_directories,
            clear_cache_files,
            MULTILINGUAL
        )
        from . import skyrimnet_chatterbox as _module
    except ImportError:
        # Fallback to absolute import (when run from PyInstaller frozen executable)
        from skyrimnet_chatterbox import (
            parse_arguments,
            set_seed,
            load_model,
            demo,
            clear_output_directories,
            clear_cache_files,
            MULTILINGUAL
        )
        import skyrimnet_chatterbox as _module

    
    args = parse_arguments()
    
    # Handle cleanup arguments that exit immediately
    if args.clearoutput:
        logger.info("Clearing output directories...")
        count = clear_output_directories()
        logger.info(f"Cleared {count} output directories. Exiting.")
        exit(0)
    
    if args.clearcache:
        logger.info("Clearing cache files...")
        count = clear_cache_files()
        logger.info(f"Cleared {count} cache files. Exiting.")
        exit(0)
    
    # Set multilingual flag
    _module.MULTILINGUAL = args.multilingual
    
    set_seed(20250527)
    model = load_model()
    
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        server_name=args.server, 
        server_port=args.port, 
        share=args.share, 
        inbrowser=args.inbrowser
    )
