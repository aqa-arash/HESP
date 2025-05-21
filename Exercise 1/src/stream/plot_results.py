import matplotlib.pyplot as plt
import pandas as pd

# Datei mit den Benchmark-Ergebnissen
file_path = "benchmark_results.txt"

# Ergebnisse einlesen, Überspringen der Kopfzeile mit Kommentaren
data = pd.read_csv(file_path, comment='/', skipinitialspace=True)

# Debugging: Spaltennamen ausgeben
print("Spaltennamen:", data.columns)

# Spaltennamen bereinigen
data.columns = data.columns.str.strip()

# Daten für Serial, OpenMP und CUDA filtern
serial_data = data[data["Version"] == "Serial"]
openmp_data = data[data["Version"] == "OpenMP"]
cuda_data = data[data["Version"] == "CUDA"]

# Puffergrößen und Bandbreiten extrahieren
buffer_sizes = serial_data["Buffer Size"]
serial_bandwidth = serial_data["Bandwidth (GBps)"]
openmp_bandwidth = openmp_data["Bandwidth (GBps)"]
cuda_bandwidth = cuda_data["Bandwidth (GBps)"]

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(buffer_sizes, serial_bandwidth, label="Serial", marker="o")
plt.plot(buffer_sizes, openmp_bandwidth, label="OpenMP", marker="s")
plt.plot(buffer_sizes, cuda_bandwidth, label="CUDA", marker="^")

# Achsenbeschriftungen und Titel
plt.xscale("log")  # Logarithmische Skala für die Puffergröße
plt.xlabel("Buffer Size (elements)", fontsize=12)
plt.ylabel("Bandwidth (GB/s)", fontsize=12)
plt.title("Bandwidth Comparison: Serial vs OpenMP vs CUDA", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Plot anzeigen
plt.tight_layout()
plt.show()