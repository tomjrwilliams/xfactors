
import sys

done = False

while not done:
    line = sys.stdin.readline().strip()
    if "module is installed, but missing library stubs" in line:
        continue
    if line.startswith("Found"):
        done = True
    print(line)