# def print_pretty_options(options):
#     print(" Gradio interface options:")
#     max_key_length = max(len(key) for key in options.keys())
#     for key, value in options.items():
#         if key == "auth" and value is not None:
#             print(f"  {key}:{' ' * (max_key_length - len(key))} {value[0]}:******")
#         else:
#             print(f"  {key}:{' ' * (max_key_length - len(key))} {value}")
#     print("")


def print_gradio_options(options):
    """
    Print Gradio server options in a compact, grouped format

    Args:
        options (dict): Dictionary of Gradio launch options
    """
    # Define parameter groups
    groups = {
        "Server": [
            "server_name",
            "server_port",
            "max_threads",
            "prevent_thread_lock",
            "quiet",
            "root_path",
            "share_server_address",
            "share_server_protocol",
            "strict_cors",
            "state_session_capacity",
            "enable_monitoring",
            "ssr_mode",
        ],
        "Display": [
            "inline",
            "inbrowser",
            "width",
            "height",
            "favicon_path",
            "show_api",
        ],
        "Access": ["share", "auth", "auth_message", "auth_dependency", "max_file_size"],
        "Security": [
            "ssl_keyfile",
            "ssl_certfile",
            "ssl_keyfile_password",
            "ssl_verify",
            "allowed_paths",
            "blocked_paths",
        ],
        "Debug": ["debug", "show_error", "app_kwargs"],
    }

    # Print header
    print(
        f"Gradio server: http://{options.get('server_name', '127.0.0.1')}:{options.get('server_port', '7770')}"
    )

    # Print each group
    for group_name, params in groups.items():
        # Filter only parameters that exist in the options dictionary
        group_params = {k: options.get(k) for k in params if k in options}

        # Skip empty groups
        if not group_params:
            continue

        # Format each parameter as "name=value"
        formatted_params = []
        for key, value in group_params.items():
            # Format the value based on its type
            if isinstance(value, str):
                if len(value) > 20:  # Truncate long strings
                    formatted_value = f'"{value[:17]}..."'
                else:
                    formatted_value = f'"{value}"'
            elif value is None:
                formatted_value = "None"
            else:
                formatted_value = str(value)

            formatted_params.append(f"{key}: {formatted_value}")

        print(f"  • {group_name.ljust(8)} > {', '.join(formatted_params)}")

    if options["server_name"] == "0.0.0.0":
        print("Notice: Server is open to the internet")
        print(
            f"Gradio server will be available on http://localhost:{options['server_port']}"
        )


# Example usage:
if __name__ == "__main__":
    # Example options
    options = {
        "inline": False,
        "inbrowser": True,
        "share": False,
        "debug": False,
        "max_threads": 40,
        "auth": None,
        "auth_message": None,
        "prevent_thread_lock": False,
        "show_error": False,
        "server_name": "127.0.0.1",
        "server_port": 7770,
        "height": "500",
        "width": "100%",
        "favicon_path": None,
        "ssl_keyfile": None,
        "ssl_certfile": None,
        "ssl_keyfile_password": None,
        "ssl_verify": True,
        "quiet": True,
        "show_api": True,
        "_frontend": True,
    }

    # Print options in a compact format
    print_gradio_options(options)
