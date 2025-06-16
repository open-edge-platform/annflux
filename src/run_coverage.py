import os
import re

from annflux.tools.mixed import get_version


def extract_test_results(input_path: str, out_path:str):
    passed_match = re.search(
        r"(\d+) passed.*?(\d+) warnings.*?(\d+:\d+:\d+)", open(input_path, "r").read()
    )
    # 1 failed, 1 passed, 1 warning in 69.41s (0:01:09)
    failed_match = re.search(
        r"(\d+) failed.*?(\d+) passed.*?(\d+) warnings.*?(\d+:\d+:\d+)", open(input_path, "r").read()
    )

    if passed_match:
        passed = int(passed_match.group(1))
        warnings = int(passed_match.group(2))
        time_taken = passed_match.group(3)
        failed = 0
    elif failed_match:
        failed = int(passed_match.group(1))
        passed = int(passed_match.group(2))
        warnings = int(passed_match.group(3))
        time_taken = passed_match.group(4)
    else:
        raise RuntimeError("Unknown test output format")

    # split time into hours, minutes, and seconds
    hours, minutes, seconds = map(int, time_taken.split(':'))

    passed_condition = hours == 0 and minutes < 5 and failed == 0

    result = {
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "time": time_taken,
        "passed_condition": passed_condition
    }
    with open(out_path, 'w') as file:
        file.write("| Metric | Value |\n")
        file.write("|--------|-------|\n")
        file.write(f"| Passed | {result['passed']} |\n")
        file.write(f"| Failed | {result['failed']} |\n")
        file.write(f"| Warnings | {result['warnings']} |\n")
        file.write(f"| Time | {result['time']} |\n")
        file.write(f"| Tests successful (Failed==0 and Time < 5:00) | {'✅' if result['passed_condition'] else '❌'} |\n")
    return result


def run_coverage_func():
    os.makedirs("../release_assets", exist_ok=True)
    os.system(
        f"ruff check . > ../release_assets/ruff_check_{get_version()['version']}.txt"
    )
    os.environ["TEST_TYPE"] = "pytest"
    if os.path.exists(".coverage"):
        print("deleting existing .coverage")
        os.remove(".coverage")
    os.system('find . -name "*.pyc" -delete')
    os.system(f"pytest annflux/tests --cov -sv > ../release_assets/tests_{get_version()['version']}.txt")
    os.system(f"coverage report -m --ignore-errors > ../release_assets/coverage_{get_version()['version']}.txt")
    extract_test_results(f"../release_assets/tests_{get_version()['version']}.txt", f"../release_assets/test_report_{get_version()['version']}.md")


if __name__ == '__main__':
    run_coverage_func()
