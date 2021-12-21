import os
import sys
from pathlib import Path


scenes = [ "benchmark_configurations/scene/conference.txt", 
           "benchmark_configurations/scene/fairyforest.txt", 
           "benchmark_configurations/scene/sibenik.txt",  
           "benchmark_configurations/scene/sanmiguel.txt" ]

setup = "benchmark_configurations/setup/benchmark_batch_50.txt"

kernels = [ "./benchmark_configurations/kernels/warmup.txt",
            "./benchmark_configurations/kernels/stackless.txt",
            "./benchmark_configurations/kernels/early_break.txt",
            "./benchmark_configurations/kernels/reference_kernels.txt", 
             ]


for scene in scenes:
    exec = "python .\\single_benchmark.py " + setup + " " + scene + " " + " ".join(kernels)
    os.system(exec)