import os
import sys
from pathlib import Path
import shutil

if len(sys.argv) < 2:
    print("Need at least one configuration files")

cameras = []
kernels = []
measurements = None
warmups = None
scene = None
exe = None
mode = None
ao_radius = None

# parse files
for idx, arg in enumerate(sys.argv):
    if idx == 0:
        continue

    with open(Path(arg)) as fp:
        Lines = fp.readlines()
        for line in Lines:
            tmp = line.split(':')
            key = tmp[0].strip()
            value = ':'.join(tmp[1:]).strip()

            if(key == "exe"):
                exe = value
            elif(key == "warmups"):
                warmups = value
            elif(key == "measurements"):
                measurements = value
            elif(key == "camera"):
                if(value[0] == '\"'):
                    cameras.append(value)
                else:
                    cameras.append('\"' + value + '\"')
            elif(key == "kernel"):
                kernels.append(value)
            elif(key == "scene"):
                scene = value
            elif(key == "mode"):
                mode = value
            elif(key == "ao-radius"):
                ao_radius = value

# check if we have all data needed
if len(cameras) == 0:
    print("Error: No camera signature specified")
    exit(-1)

if len(kernels) == 0:
    print("Error: No tracing kernels specified")
    exit(-1)

if exe == None or exe == "":
    print("Error: No executable specified")
    exit(-1)

if scene == None or scene == "":
    print("Error: No scene specified")
    exit(-1)

if warmups == None:
    print("Error: Number of warmup runs is not specified")
    exit(-1)    

if measurements == None:
    print("Error: Number of measurements runs is not specified")
    exit(-1)    

if mode == None:
    print("Error: Benchmarking mode is not specified")
    exit(-1)  

kernel_prefix = "--kernel="
camera_prefix = "--camera="
mode_prefix = "--mode="
warmup_prefix = "--warmup-repeats="
measure_prefix = "--measure-repeats="
scene_prefix = "--mesh="
ao_radius_prefix = "--ao-radius="


exe_args = scene_prefix + scene
exe_args += " "

exe_args += mode_prefix + mode
exe_args += " "

exe_args += warmup_prefix + warmups
exe_args += " "

exe_args += measure_prefix + measurements
exe_args += " "

if ao_radius is not None:
    exe_args += ao_radius_prefix + ao_radius
    exe_args += " "

for camera in cameras:
    exe_args += camera_prefix + camera
    exe_args += " "

for kernel in kernels:
    exe_args += kernel_prefix + kernel
    exe_args += " "


exe_string = exe + " " + "benchmark" + " " + exe_args



# clean cudacache
os.system("rmdir /s /q cudacache")
os.system("rmdir /s /q benchmarks\out")
os.system("mkdir benchmarks\out")

os.system(exe_string)
src_folder = Path("benchmarks/out")
dst_folder = Path("benchmarks/data")
for file_name in os.listdir(src_folder):
    src = os.path.join(src_folder, file_name)
    dst = os.path.join(dst_folder, file_name)
    if os.path.isfile(src):
        shutil.copy(src, dst)
