import subprocess
import sys
import pkg_resources


# Installs the specified package using pip
def install_package(package_name):
    subprocess.check_call([sys.executable,
                          "-m", "pip", "install", package_name])


# Checks if the specified package is installed
def is_package_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False


# Installs packages listed in the specified file
def setup(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            package_name = line.strip()
            if not is_package_installed(package_name):
                install_package(package_name)


setup("requirements.txt")
setup("requirements-dev.txt")

print("setup finished")
