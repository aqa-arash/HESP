import matplotlib.pyplot as plt
from collections import defaultdict
import re

# Map config name to particle number
def get_particles(config):
    if "2k5" in config:
        return 2500
    elif "5k" in config:
        return 5000
    elif "10k" in config:
        return 10000
    else:
        return 0

# Map config name to group
def get_group(config):
    if "_co_acc" in config:
        return "acc"
    elif "_co" in config:
        return "co"
    else:
        return "base"

# Read runtimes.txt
data = defaultdict(list)
with open("runtimes.txt") as f:
    for line in f:
        m = re.match(r"(config[0-9a-zA-Z_]+\.txt):\s*([0-9.]+)", line)
        if m:
            config = m.group(1).replace(".txt", "")
            time = float(m.group(2))
            data[config].append(time)

# Compute averages
averages = defaultdict(dict)
for config, times in data.items():
    particles = get_particles(config)
    group = get_group(config)
    avg = sum(times) / len(times)
    averages[group][particles] = avg

# Plot
plt.figure(figsize=(10,6))
markers = {'acc': 'o-', 'co': 's-', 'base': '^-'}
labels = {'acc': 'acceleration', 'co': 'cutoff', 'base': 'base'}

for group in ['acc', 'co', 'base']:
    x = sorted(averages[group].keys())
    y = [averages[group][p] for p in x]
    if x:
        plt.plot(x, y, markers[group], label=labels[group])

plt.title("Average Runtime over Particle Number on Nvidia GeForce GTX 965M ")
plt.xlabel("Particle Number (log scale)")
plt.ylabel("Average Runtime (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xscale("log")
plt.savefig("avg_runtimes.png")
plt.show()