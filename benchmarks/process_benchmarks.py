import csv
import sys
import matplotlib.pyplot as plt
import numpy as np

def process_header(header_row):
    metadata = {}
    metadata["obj_file"] = header_row[1].split('/')[-1]
    metadata["num_kernels"] = int(header_row[2])
    metadata["num_cameras"] = int(header_row[3])
    metadata["num_rays"] = int(header_row[4])
    metadata["num_warmup"] = int(header_row[5])
    metadata["num_measure"] = int(header_row[6])
    return metadata

def init_data(info, data):
    measurements = info["num_warmup"] + info["num_measure"]

    for kernel_id in range(info["num_kernels"]):
        data.append([])
        for measure_id in range(measurements):
            data[kernel_id].append([])
            for ray_id in range(info["num_rays"]):
                data[kernel_id][measure_id].append([])
                for camera_id in range(info["num_cameras"]):
                    data[kernel_id][measure_id][ray_id].append({})
                    data[kernel_id][measure_id][ray_id][camera_id]["time"] = 3.0
                    data[kernel_id][measure_id][ray_id][camera_id]["rays"] = 4
    
    

def print_raw_data(info, data, kernels, rays):
    measurements = info["num_warmup"] + info["num_measure"]

    for kernel_id in range(info["num_kernels"]):
        for measure_id in range(measurements):
            for ray_id in range(info["num_rays"]):
                for camera_id in range(info["num_cameras"]):
                    print("%s %d %s %d %f %d" % (kernels[kernel_id], measure_id, rays[ray_id], camera_id, data[kernel_id][measure_id][ray_id][camera_id]["time"], data[kernel_id][measure_id][ray_id][camera_id]["rays"]) )



def get_mrays_per_measurement(info, data, kernels, rays, drop_warmup):
    mray_data = []
    measurements = info["num_warmup"] + info["num_measure"]
    for kernel_id in range(info["num_kernels"]):
        mray_data.append([])
        mid = 0
        for measure_id in range(measurements):
            if drop_warmup and (measure_id < info["num_warmup"]):
                continue
            mray_data[kernel_id].append([])
            #print("\n%s %d " % (kernels[kernel_id], measure_id),end='')
            for ray_id in range(info["num_rays"]):
                mray_data[kernel_id][mid].append([])
                total_time = 0.0
                total_rays = 0
                for camera_id in range(info["num_cameras"]):
                    total_time += data[kernel_id][measure_id][ray_id][camera_id]["time"]
                    total_rays += data[kernel_id][measure_id][ray_id][camera_id]["rays"]
                    
                mrays = total_rays / total_time * 10**(-6)
                mray_data[kernel_id][mid][ray_id] = mrays
                #print("\t%f" % mrays, end='')
            mid += 1
                
    return mray_data

def print_mrays_per_measurement(info, mray_data, kernels, rays):
    for kernel_id in range(len(mray_data)):
        for measure_id in range(len(mray_data[kernel_id])):
            print("\n%s %d " % (kernels[kernel_id], measure_id),end='')
            for ray_id in range(len(mray_data[kernel_id][measure_id])):
                mrays = mray_data[kernel_id][measure_id][ray_id]
                print("\t%f" % mrays, end='')

def get_box_plot_data(mray_data, ray_type):
    data = []
    for kernel_id in range(len(mray_data)):
        d = []
        for measure_id in range(len(mray_data[kernel_id])):
            mrays = mray_data[kernel_id][measure_id][ray_type]
            d.append(mrays)
        data.append(np.array(d))
    return data



# ================== main =====================

print('Number of arguments:' + str(len(sys.argv)))
print('Argument List:', str(sys.argv))

if len(sys.argv) < 2:
    print("To few arguments: Need at least a file to process\n")

benchmark_info = None
data = []
kernels = []
rays = []

with open(sys.argv[1], newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row[0] == '#header':
            benchmark_info = process_header(row)
            init_data(benchmark_info, data)
            print(benchmark_info)
            
        else:
            kernel = row[0].strip()
            if kernel not in kernels:
                kernels.append(kernel)
            kernel_index = kernels.index(kernel)

            measurment = int(row[1])

            ray_type = row[2].strip()
            if ray_type not in rays:
                rays.append(ray_type)
            ray_index = rays.index(ray_type)

            camera = int(row[3])
            time_mes = float(row[4])
            rays_mes = int(row[5])

            data[kernel_index][measurment][ray_index][camera]["time"] = time_mes
            data[kernel_index][measurment][ray_index][camera]["rays"] = rays_mes

            #print(' '.join(row))

print("\n\n")
print_raw_data(benchmark_info, data, kernels, rays)
mray_data = get_mrays_per_measurement(benchmark_info, data, kernels, rays, True)
print_mrays_per_measurement(benchmark_info, mray_data, kernels, rays)

np.random.seed(10)
data = get_box_plot_data(mray_data, 0)
 
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(data)
plt.xticks([1, 2], kernels)
 
# show plot
plt.show()

