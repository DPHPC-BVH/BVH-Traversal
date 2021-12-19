import csv
import sys
import matplotlib.pyplot as plt
from scipy.stats.stats import _equal_var_ttest_denom
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
import numpy as np
import argparse
import statistics
import math

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

def get_time_per_measurement_ms(info, data, drop_warmup):
    time_data = []
    measurements = info["num_warmup"] + info["num_measure"]
    for kernel_id in range(info["num_kernels"]):
        time_data.append([])
        mid = 0
        for measure_id in range(measurements):
            if drop_warmup and (measure_id < info["num_warmup"]):
                continue
            time_data[kernel_id].append([])
            #print("\n%s %d " % (kernels[kernel_id], measure_id),end='')
            for ray_id in range(info["num_rays"]):
                time_data[kernel_id][mid].append([])
                total_time = 0.0
                for camera_id in range(info["num_cameras"]):
                    total_time += data[kernel_id][measure_id][ray_id][camera_id]["time"]  
                time_data[kernel_id][mid][ray_id] = total_time*1000
                #print("\t%f" % mrays, end='')
            mid += 1
                
    return time_data

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

# output kernels: list of kernels for which data should e returned
# data is returned in the same order as defined by output_kernels
def get_plot_data(data, ray_type, all_kernels, output_kernels):
    data = []
    for kernel in output_kernels:
        d = []
        kernel_id = all_kernels.index(kernel)
        for measure_id in range(len(mray_data[kernel_id])):
            mrays = mray_data[kernel_id][measure_id][ray_type]
            d.append(mrays)
        data.append(np.array(d))
    return data

def get_plot_cnf(path):
    kernel_list = []
    k = {}
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if row[0][0] == '#':
                pass
            else:
                kernel = row[0].strip()
                label = row[1].strip()
                ref = row[2].strip()
                kernel_list.append(kernel)
                k[kernel] = {}
                k[kernel]["label"] = label
                k[kernel]["is_reference"] = ref == "reference"

    return kernel_list, k

# returns a list of labels for kernels
def get_kernel_label_list(kernel_metadata, kernels):
    labels = []
    for kernel in kernels:    
        if kernel in kernel_metadata:
            labels.append(kernel_metadata[kernel]["label"])
        else:
            labels.append(kernel)
    return labels

def is_reference_kernel(kernel_metadata, kernel):
    if kernel in kernel_metadata:
        return kernel_metadata[kernel]["is_reference"]
    else:
        return False
    

def get_median_95ci(data):
    sorted_data = np.sort(data)
    #print(sorted_data)
    n = sorted_data.size
    
    low = math.floor((n-1.96*np.sqrt(n))/2)
    high = math.ceil(1+(n+1.96*np.sqrt(n))/2)

    return sorted_data[low], sorted_data[high]


def largest_diff_95ci(data_array):
    diff = 0
    for data in data_array:
        #print(data)
        low, high = get_median_95ci(data)
        median = np.median(data)
        diff_low = abs(median-low)/median
        diff_high = abs(high-median)/median
        diff = max(diff, diff_low)
        diff = max(diff, diff_high)
    return diff


def largest_stddev(data_array):
    stddev = 0
    for data in data_array:
        pass





# ================== main =====================

#print('Number of arguments:' + str(len(sys.argv)))
#print('Argument List:', str(sys.argv))

if len(sys.argv) < 2:
    print("To few arguments: Need at least a file to process\n")

# parse arguments
parser = argparse.ArgumentParser(description='Process a raw benchmark_file')
parser.add_argument('src', help='Path to the raw benchmark data')
parser.add_argument('--normal_test', action="store_true", help='perform a normality test on all the sample sets and raises an alert if one ore multiple are not normally distributed')
parser.add_argument('--median_ci_test', action="store_true", help='calculate nonparametric CIs for all sample sets and checks if they are within 5\% of the median')
parser.add_argument('--box_plot', action="store_true", help='make boxplots for all the data sets')
parser.add_argument('--QQ_plot', action="store_true", help='make QQ plots for all the data sets')
parser.add_argument('--dump_table', action="store_true", help='dump a table with the summarized measurements for each kernel and ray type')
parser.add_argument('--time', action="store_true", help='do calculations using time measurements only')
parser.add_argument('--no_title', action="store_true", help='Remove Titels from the plots')
parser.add_argument('--xfmt', action="store_true", help='Automatically rotate labels on the x-axis to prevent overlaps')
parser.add_argument('--plot_cnf', action="store", help='path to the plot config file')
parser.add_argument('--plot_stat_data', action="store_true", help='adds an aditional box with statistical information to the plot')
parser.add_argument('--bar_plot', action="store_true", help='makes bar plots')

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


if args.plot_cnf is None:
    kernel_plot_metadata = {}
    output_kernels = kernels.copy()
else:
    k_list, kernel_plot_metadata = get_plot_cnf(args.plot_cnf)
    output_kernels = [x for x in kernels if x in k_list]

print("\n\n")
#print_raw_data(benchmark_info, data, kernels, rays)
if args.time:
    mray_data = get_time_per_measurement_ms(benchmark_info, data, True)
else:
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
            norm_test_log += ("%s %s \t p=%.4f" %(kernel, ray_types[i], p))
            if p < 0.05:
                norm_test_log += "\t ***THRESHOLD NOT REACHED***"
            norm_test_log += "\n"
            #print('%s %s Statistics=%.3f, p=%.3f' % (kernel, ray_types[i], stat, p))
    file_name = scene_name + "_" + str(benchmark_info["num_measure"]) + "_normtest" + ".txt"
    file_path = test_out_path + "/" + file_name
    with open(file_path, "w") as o_file:
        o_file.write("%s" % norm_test_log)
    
            

if args.QQ_plot:
    for i in range(len(rays)):
        data = get_box_plot_data(mray_data, i)
        for idx, kernel in enumerate(kernels):
            qqplot(data[idx], line='s')
            plt.savefig(out_path + 'QQ/' + scene_name + "_" + kernel + "_" + ray_types[i] + "_" + str(benchmark_info["num_measure"]) + ".png")


if args.box_plot or args.bar_plot:
    label_list = get_kernel_label_list(kernel_plot_metadata, output_kernels)
    for i in range(len(rays)):
        data = get_plot_data(data, i, kernels, output_kernels)
        #data = get_box_plot_data(mray_data, i)
        plt.figure(figsize =(10, 7))
        fig, ax = plt.subplots(figsize = (10, 7))

        #plt.figure()
        #fig, ax = plt.subplots()

        #plt.figure(figsize =(15, 10.5))
        #fig, ax = plt.subplots(figsize = (15, 10.5))
        # Creating plot
        
        boxprops = dict(linestyle='--', linewidth=1.5)
        medianprops = dict(linewidth=1.5)
        bplot = plt.boxplot(data, notch=True, labels=label_list, patch_artist=True, boxprops=boxprops, medianprops=medianprops)

        

        if args.time:
            y_label = plt.ylabel('Runtime [ms]', fontsize=16, labelpad=10)
        else:
            y_label = plt.ylabel('Performance [Mrays / s]', fontsize=16, labelpad=10)

        #y_label.set_rotation(0)

        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)


        # set title plot
        title = "Benchmark " + ray_types[i] + " Rays for " + scene_name + " with " + str(benchmark_info["num_cameras"]) + " cameras" 

        if not args.no_title:
            plt.title(title, fontsize=20, pad=15)

        # set colors
        for idx, box in enumerate(bplot['boxes']):
            if is_reference_kernel(kernel_plot_metadata, output_kernels[idx]):
                box.set_facecolor('lightblue')
            else:
                box.set_facecolor('lightgreen')

        # set border
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)

        # set grid
        ax.yaxis.grid()

        # format x-labels
        if args.xfmt:
            fig.autofmt_xdate()

        # add statistical data
        if args.plot_stat_data:
            text_props = dict(boxstyle='round', facecolor='wheat')

            diff = largest_diff_95ci(data)
            ci_diff = "%.2f" % (diff*100)
            stat_str = r'$95 \% \ CI: \pm$ ' + ci_diff + r'$\%$' 

            # place a text box in upper left in axes coords
            ax.text(0.95, 0.95, stat_str, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=text_props)  
                
        fig.tight_layout()
        
        plt.savefig(out_path + scene_name + "_" + ray_types[i] + "_" + str(benchmark_info["num_measure"]) + ".png")
        #plt.show()

if args.dump_table:
    data = []
    for i in range(len(rays)):
        data.append(get_box_plot_data(mray_data, i))

    measurement_dump = ""

    for idx, kernel in enumerate(kernels):
        for i in range(len(rays)):
            d = data[i][idx]
            var = statistics.variance(d)
            mean = statistics.mean(d)
            std_dev = statistics.stdev(d)
            measurement_dump += ("%s %s [Mrays/s] \t mean: %f var: %f std_deviation: %f\n" % (kernel, ray_types[i], mean, var, std_dev))
            #print("%s %s [Mrays/s]:" % (kernel, ray_types[i]))
            
            for j, mes in enumerate(d):
                #print("\t %.2f" % (mes))
                measurement_dump += ("  (%d)\t %.2f\n" % (j+1, mes))

    file_name = scene_name + "_" + str(benchmark_info["num_measure"]) + "_dump" + ".txt"
    file_path = 'benchmarks/dump/' + file_name
    with open(file_path, "w") as o_file:
        o_file.write("%s" % measurement_dump)
    
            


