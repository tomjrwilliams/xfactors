
import sys

done = False

SKIP = [
    "module is installed, but missing library stubs",
    "See https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports",
    "Cannot find implementation or library stub for module named \"jaxlib.",
]

while not done:
    line = sys.stdin.readline().strip()
    if any([k in line for k in SKIP]):
        continue
    if line.startswith("Found"):
        done = True
    print(line)