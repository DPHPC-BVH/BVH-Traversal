# Running single benchmarks from configuration files

To run a single benchmark from a given configuration execute:

```
python single_benchmark.py <config_1> <config_2> ...
```

Example:
```
python single_benchmark.py ./benchmark_configurations/scene/scene_template.txt ./benchmark_configurations/setup/setup_template.txt ./benchmark_configurations/kernels/test_kernels.txt
```

### Note: it does not matter where the arguments are specified. To run a successfull benchmark however, the following parameters have to be specified in at least on of the configuration files:

- exe
- warmups
- measurements
- scene
- camera (at least one)
- kernel (at least one)
- mode


### Note: you can specify the path to a configuration file in both unix "/" and windows "\\" format.

# Processing Mode 2 Benchmarks


To parse and automatically generate box plots of a mode 2 benchmark execute:
```
python benchmarks/process_benchmarks.py <data_file> --box_plot
```
where *<data_file>* is the path to the raw benchmark data dumped by the ray tracer.

To perform a normality test for all sample sets of a mode 2 benchmark execute:
```
python benchmarks/process_benchmarks.py <data_file> --normal_test
```

To draw QQ plots for all sample sets of a mode 2 benchmark execute:
```
python benchmarks/process_benchmarks.py <data_file> --QQ_plot
```
boxplots are placed into benchmarks/img \
QQ plots are placed into benchmarks/img/QQ \
normality test results are placed into benchmarks/norm_tests 

for more information and options execute
```
python benchmarks/process_benchmarks.py -h
```

---
**IMPORTATNT NOTE**

**To work correctly the commands have to be executed in the main directory of the project**

Mode 2 benchmark dumps are put into the ./benchmarks/out folder by the tracer.

If a dump fails to be overwritten by the tracer, remove the already existing dump of the name for the out folder.

---
