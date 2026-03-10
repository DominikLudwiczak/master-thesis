BLOCKED = [
    "rm -rf /",
    "shutdown",
    "reboot",
    "mkfs",
    "sudo"
]


def safe(cmd):
    for b in BLOCKED:
        if b in cmd:
            return False
    return True