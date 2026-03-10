import json
from tools import run_command, read_file, list_dir
from safety import safe
from llm import ask_llm, SYSTEM_PROMPT


class RepoAgent:

    def __init__(self, repo_path):

        self.repo = repo_path

        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def observe(self, text):
        self.messages.append(
            {"role": "user", "content": text}
        )

    def think(self):

        reply = ask_llm(self.messages)

        self.messages.append(
            {"role": "assistant", "content": reply}
        )

        return json.loads(reply)

    def act(self, action):

        t = action["action"]
        args = action.get("args", {})

        if t == "run":

            cmd = args["cmd"]

            if not safe(cmd):
                return "Command blocked for safety"

            code, out = run_command(cmd, cwd=self.repo)

            return f"exit={code}\n{out}"

        if t == "read":

            path = args["path"]

            return read_file(f"{self.repo}/{path}")

        if t == "list":

            path = args.get("path", ".")

            return str(list_dir(f"{self.repo}/{path}"))

        if t == "finish":

            return "FINISHED"

        return "unknown action"

    def run(self):

        self.observe(
            "Start analyzing repository. Read README.md first."
        )

        for _ in range(30):

            action = self.think()

            result = self.act(action)

            if result == "FINISHED":
                print("Agent finished")
                break

            self.observe(result)