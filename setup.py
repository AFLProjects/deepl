import subprocess
import sys
import pkg_resources

def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def is_package_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

with open('requirements.txt', 'r') as file:
    for line in file:
        package_name = line.strip()
        if not is_package_installed(package_name):
            install_package(package_name)

print("setup finished")