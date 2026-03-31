import subprocess
from agent import RepoAgent


def clone(repo_url):
    subprocess.run(
        f"rm -r workspace",
        shell=True
    )

    subprocess.run(
        f"git clone {repo_url} workspace",
        shell=True
    )


def main():

    repo = "https://github.com/coinse/fonte"

    # clone(repo)

    agent = RepoAgent("workspace")

    agent.run()


if __name__ == "__main__":
    main()