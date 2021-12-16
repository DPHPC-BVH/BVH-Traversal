import csv
import sys
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
import numpy as np
import argparse

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
        k_a = []
        for measure_id in range(measurements):
            m_a = []
            for ray_id in range(info["num_rays"]):
                r_a = []
                for camera_id in range(info["num_cameras"]):
                    d = {}
                    d["time"] = 3.0
                    d["rays"] = 4
                    r_a.append(d)
                m_a.append(r_a)
            k_a.append(m_a)
        data.append(k_a)
    

def print_raw_data(info, data, kernels, rays):
    measurements = info["num_warmup"] + info["num_measure"]

    for kernel_id in range(info["num_kernels"]):
        k_a = data[kernel_id]
        for measure_id in range(measurements):
            m_a = k_a[measure_id]
            for ray_id in range(info["num_rays"]):
                r_a = m_a[ray_id]
                for camera_id in range(info["num_cameras"]):
                    c_entry = r_a[camera_id]
                    print("%s %d %s %d %f %d" % (kernels[kernel_id], measure_id, rays[ray_id], camera_id, c_entry["time"], c_entry["rays"]) )



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
            #print("\n%s %d " % (kernels[kernel_id], measure_id),end='')
            for ray_id in range(len(mray_data[kernel_id][measure_id])):
                mrays = mray_data[kernel_id][measure_id][ray_id]
               #print("\t%f" % mrays, end='')

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

#print('Number of arguments:' + str(len(sys.argv)))
#print('Argument List:', str(sys.argv))

if len(sys.argv) < 2:
    print("To few arguments: Need at least a file to process\n")

# parse arguments
parser = argparse.ArgumentParser(description='Process a raw benchmark_file')
parser.add_argument('src', help='Path to the raw benchmark data')
parser.add_argument('--normal_test', action="store_true", help='perform a normality test on all the sample sets and raises an alert if one ore multiple are not normally distributed')
parser.add_argument('--box_plot', action="store_true", help='make boxplots for all the data sets')
parser.add_argument('--QQ_plot', action="store_true", help='make QQ plots for all the data sets')

args = parser.parse_args()


benchmark_info = None
data = []
kernels = []
rays = []
ray_types = ["Primary", "AO", "Diffuse"]

with open(args.src, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row[0] == '#header':
            benchmark_info = process_header(row)
            init_data(benchmark_info, data)
            #print(benchmark_info)
            
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
#print_raw_data(benchmark_info, data, kernels, rays)
mray_data = get_mrays_per_measurement(benchmark_info, data, kernels, rays, True)
print_mrays_per_measurement(benchmark_info, mray_data, kernels, rays)
print("\n")
#np.random.seed(10)
# returns a list containing a np.array for each kernel with all Mray/s measurements as float
# the second parameter decides which ray type should be extracted: 0=>Primary, 1=>AO, 2=>diffuse

scene_name = benchmark_info["obj_file"].split('.')[0]
out_path = 'benchmarks/img/'
test_out_path = 'benchmarks/norm_tests'

if args.normal_test:
    norm_test_log = ""
    for i in range(len(rays)):
        data = get_box_plot_data(mray_data, i)
        for idx, kernel in enumerate(kernels):
            stat, p = shapiro(data[idx])
            norm_test_log += ("%s %s \t p=%.4f" %(kernel, ray_types[i], p)) + "\n"
            #print('%s %s Statistics=%.3f, p=%.3f' % (kernel, ray_types[i], stat, p))
    file_name = scene_name + "_" + str(benchmark_info["num_measure"]) + ".txt"
    file_path = test_out_path + "/" + file_name
    with open(file_path, "w") as o_file:
        o_file.write("%s" % norm_test_log)
    
            

if args.QQ_plot:
    for i in range(len(rays)):
        data = get_box_plot_data(mray_data, i)
        for idx, kernel in enumerate(kernels):
            qqplot(data[idx], line='s')
            plt.savefig(out_path + 'QQ/' + scene_name + "_" + kernel + "_" + ray_types[i] + "_" + str(benchmark_info["num_measure"]) + ".png")


if args.box_plot:
    for i in range(len(rays)):
        data = get_box_plot_data(mray_data, i)
        fig = plt.figure(figsize =(10, 7))
 
        # Creating plot
        plt.boxplot(data, notch=True)

        plt.xticks(list(range(1, len(kernels)+1)), kernels)
 
        # show plot
        title = "Benchmark " + ray_types[i] + " Rays for " + scene_name + " with " + str(benchmark_info["num_cameras"]) + " cameras" 
        plt.title(title)
        
        plt.savefig(out_path + scene_name + "_" + ray_types[i] + "_" + str(benchmark_info["num_measure"]) + ".png")
        #plt.show()

