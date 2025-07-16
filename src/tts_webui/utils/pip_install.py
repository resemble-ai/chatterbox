import os
import subprocess


def write_log(output, name, type):
    script_dir = os.path.dirname((__file__))
    with open(
        os.path.join(script_dir, "..", "..", "logs", f"{type}-{name}.log"), "w"
    ) as outfile:
        outfile.write("\n".join(output))


def pip_install_wrapper(requirements, name, include_gradio=True):
    def fn():
        output = []
        for line in _pip_install(requirements + " gradio==5.5.0" if include_gradio else "", name):
            output.append(str(line))
            yield "<br />".join(output)

        write_log(output, name, type="pip-install")
        return line
        # verify installation by importing the package or drying run install

    return fn


def pip_uninstall_wrapper(package_name, name):
    def fn():
        output = []
        for line in _pip_uninstall(package_name, name):
            output.append(line)
            yield "<br />".join(output)

        write_log(output, name, type="pip-uninstall")

    return fn


def _pip_install(requirements, name):
    # process = subprocess.Popen(
    #     f"uv pip install {requirements}",
    #     shell=True,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.STDOUT,
    #     universal_newlines=True,
    # )

    # # Stream the output to the console
    # for line in process.stdout:  # type: ignore
    #     print(line, end="")
    #     yield line

    # # Wait for the process to finish
    # process.wait()

    # # Check if the process was successful
    # if process.returncode == 0:
    #     print(f"Successfully installed {name}")
    #     yield f"Successfully installed {name}, please restart the webui"
    # else:
    #     print(f"Failed to install {name}")
    #     yield f"Failed to install {name}"
    try:
        print(f"Installing {name} dependencies...")
        yield from pip_shell(f"pip install {requirements}")
        # yield from pip_shell(f"uv pip install {requirements}")
        print(f"Successfully installed {name} dependencies")
        yield f"Successfully installed {name} dependencies"
        yield f"Please restart the webui to see the changes"
        yield True
    except Exception:
        print(f"Failed to install {name} dependencies")
        yield f"Failed to install {name} dependencies"
        yield False


def _pip_uninstall(package_name, name):
    try:
        print(f"Uninstalling {name} ({package_name})...")
        yield from pip_shell(f"pip uninstall -y {package_name}")
        # yield from pip_shell(f"uv pip uninstall {package_name}")
        print(f"Successfully uninstalled {name} ({package_name})")
        yield f"Successfully uninstalled {name} ({package_name})"
    except Exception:
        print(f"Failed to uninstall {name} ({package_name})")
        yield f"Failed to uninstall {name} ({package_name})"


def pip_shell(command):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    # Stream the output to the console
    for line in process.stdout:  # type: ignore
        print(line, end="")
        yield line

    # Wait for the process to finish
    process.wait()

    if process.returncode != 0:
        raise Exception(f"Failed to run {command}")
