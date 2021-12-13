# Running single benchmarks from configuration files

To run a single benchmark from a given configuration execute:

```
python single_benchmark.py <config_1> <config_2> ...
```

Example:
```
python single_benchmark.py ./benchmark_configurations/scene/scene_template.txt ./benchmark_configurations/setup/setup_template.txt
```

### Note: it does not matter where the arguments are specified. To run a successfull benchmark however, the following parameters have to be specified:

- exe
- warmups
- measurements
- scene
- camera (at least one)
- kernel (at least one)
- mode


### Note: you can specify the path to a configuration file in both unix "/" and windows "\\" format.

# Processing Mode 2 Benchmarks

To parse and automatically generate a box plot of a mode 2 benchmark execute:
```
python benchmarks/process_benchmarks.py <output_file>
```
where *<output_file>* is the path to the raw benchmark data dumped by the ray tracer.

### Note: Mode 2 benchmark dumps are put into the ./benchmarks/out folder by the tracer.

### Note: If a dump fails to be overwritten by the tracer, remove the already existing dump of the name for the out folder.
