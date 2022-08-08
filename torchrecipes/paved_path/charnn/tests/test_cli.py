import os
import subprocess

WORKING_DIR = os.path.dirname(os.path.dirname(__file__))


def run(command):
    """Run a command and return error code"""
    proc = subprocess.Popen(command)
    proc.communicate()
    return proc.returncode


def test_main():
    command = ["python3", os.path.join(WORKING_DIR, "main.py")]
    assert run(command) == 0
