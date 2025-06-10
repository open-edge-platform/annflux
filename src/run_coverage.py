import os

from annflux.ui.basic.run_server import get_version


def run_coverage_func():
    os.system(
        f"ruff check . > ../release_assets/ruff_check_{get_version()['version']}.txt"
    )
    os.environ["TEST_TYPE"] = "pytest"
    if os.path.exists(".coverage"):
        print("deleting existing .coverage")
        os.remove(".coverage")
    os.system('find . -name "*.pyc" -delete')
    os.system("pytest annflux/tests --cov -sv")
    os.system(f"coverage report -m --ignore-errors > ../release_assets/coverage_{get_version()['version']}.txt")


if __name__ == '__main__':
    run_coverage_func()