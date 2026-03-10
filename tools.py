import subprocess
from pathlib import Path


def run_command(cmd, cwd=None):
    p = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    output = p.stdout.splitlines()[-50:]
    return p.returncode, "\n".join(output)


def read_file(path):
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text()[:15000]


def list_dir(path):
    p = Path(path)
    if not p.exists():
        return []
    return [x.name for x in p.iterdir()]


def write_file(path, content):
    Path(path).write_text(content)