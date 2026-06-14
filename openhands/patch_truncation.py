"""Patch OpenHands truncate_content() to use asymmetric head+mid+tail split.

Run inside the Docker build:
    COPY patch_truncation.py /tmp/
    RUN python3 /tmp/patch_truncation.py
"""
import pathlib

TARGET = pathlib.Path("/app/openhands/events/serialization/event.py")

# The exact original function from OpenHands 1.4
OLD_FUNCTION = """def truncate_content(content: str, max_chars: int | None = None) -> str:
    \"\"\"Truncate the middle of the observation content if it is too long.\"\"\"
    if max_chars is None or len(content) <= max_chars or max_chars < 0:
        return content

    # truncate the middle and include a message to the LLM about it
    half = max_chars // 2
    return (
        content[:half]
        + '\\n[... Observation truncated due to length ...]\\n'
        + content[-half:]
    )"""

NEW_FUNCTION = """def truncate_content(content: str, max_chars: int | None = None) -> str:
    \"\"\"Truncate observation content with asymmetric head+mid+tail split.

    Favours the tail (where errors/results appear) over the head.
    Split ratio: ~14% head, ~29% middle, ~57% tail.
    \"\"\"
    if max_chars is None or len(content) <= max_chars or max_chars < 0:
        return content

    head_size = max(200, max_chars * 14 // 100)
    mid_size = max(200, max_chars * 29 // 100)
    tail_size = max_chars - head_size - mid_size

    mid_start = (len(content) - mid_size) // 2

    return (
        content[:head_size]
        + '\\n[... truncated ...]\\n'
        + content[mid_start:mid_start + mid_size]
        + '\\n[... truncated ...]\\n'
        + content[-tail_size:]
    )"""

src = TARGET.read_text()
if OLD_FUNCTION not in src:
    print("WARNING: exact old function not found, trying relaxed match...")
    # Try matching just the function signature + half logic
    if "half = max_chars // 2" in src:
        # Replace just the body
        src = src.replace(
            "half = max_chars // 2\n"
            "    return (\n"
            "        content[:half]\n"
            "        + '\\n[... Observation truncated due to length ...]\\n'\n"
            "        + content[-half:]\n"
            "    )",
            "head_size = max(200, max_chars * 14 // 100)\n"
            "    mid_size = max(200, max_chars * 29 // 100)\n"
            "    tail_size = max_chars - head_size - mid_size\n"
            "    mid_start = (len(content) - mid_size) // 2\n"
            "    return (\n"
            "        content[:head_size]\n"
            "        + '\\n[... truncated ...]\\n'\n"
            "        + content[mid_start:mid_start + mid_size]\n"
            "        + '\\n[... truncated ...]\\n'\n"
            "        + content[-tail_size:]\n"
            "    )",
        )
        TARGET.write_text(src)
        print("Patched truncate_content() (relaxed match)")
    else:
        raise RuntimeError("Could not find truncate_content() to patch")
else:
    src = src.replace(OLD_FUNCTION, NEW_FUNCTION)
    TARGET.write_text(src)
    print("Patched truncate_content() (exact match)")
