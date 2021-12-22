import csv
from os import error
import sys
import matplotlib.pyplot as plt
from scipy.stats.stats import _equal_var_ttest_denom
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
import scipy
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

# output kernels: list of kernels for which data should e returned
# data is returned in the same order as defined by output_kernels
def get_plot_data(p_data, ray_type, all_kernels, output_kernels):
 
    data = []
    for kernel in output_kernels:
        d = []
        kernel_id = all_kernels.index(kernel)
        for measure_id in range(len(p_data[kernel_id])):
            mrays = p_data[kernel_id][measure_id][ray_type]
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

def get_optix_measurements(path, scene_id):
    data = None
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0].strip() == '#header' or not (row[0].split("/")[-1].strip()  == scene_id.strip()):
                pass
            else:
                num_measurement = len(row) - 2
                if data == None:
                    data = [0.0] * num_measurement
                for i in range(num_measurement):
                    # assume measurements are in us
                    data[i] += float(row[i+2].strip())/1000

    print(data)
    return np.array(data)


def get_primary_plus_ao_time(data, kernels, output_kernels):
    primary_data = get_plot_data(data, 0, kernels, output_kernels)
    ao_data = get_plot_data(data, 1, kernels, output_kernels)

    optix_data = []
    for idx in range(len(primary_data)):
        primary_mes = primary_data[idx]
        ao_mes = ao_data[idx]
        optix_mes = np.add(primary_mes, ao_mes)
        optix_data.append(optix_mes)

    return optix_data


def get_medians(data):
    medians = []
    for measurements in data:
        m = np.median(measurements)
        medians.append(m)
    return medians

def get_arithmetic_means(data):
    means = []
    for measurements in data:
        m = np.mean(measurements)
        means.append(m)
    return means


def mean_confidence_interval_error(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print ("mean, error: %f %f" %(m, h))
    error = h/m
    return error

def mean_largest_diff_95ci(data_array):
    diff = 0
    for data in data_array:
        err = mean_confidence_interval_error(data)
        diff = max(diff,err)
    return diff

def mean_95ci_errors(data_array):
    errors = []
    for data in data_array:
        errors.append(mean_confidence_interval_error(data))
    return errors

def median_95ci_errors(data_array):
    min = []
    max = []
    for data in data_array:
        median = np.median(data)
        low, high  = get_median_95ci(data)
        min.append(median - low)
        max.append(high - median)
    return [min,max]

        
def read_benchmark_file(path):
    b_info = None
    b_data = []
    b_kernels = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[0] == '#header':
                b_info = process_header(row)
                init_data(b_info, b_data)
                #print(benchmark_info)
            
            else:
                kernel = row[0].strip()
                if kernel not in b_kernels:
                    b_kernels.append(kernel)
                kernel_index = b_kernels.index(kernel)

                measurment = int(row[1])

                ray_type = row[2].strip()
                if ray_type not in rays:
                    rays.append(ray_type)
                ray_index = rays.index(ray_type)

                camera = int(row[3])
                time_mes = float(row[4])
                rays_mes = int(row[5])

                b_data[kernel_index][measurment][ray_index][camera]["time"] = time_mes
                b_data[kernel_index][measurment][ray_index][camera]["rays"] = rays_mes

    return b_info, b_kernels, b_data

#def get_optix_time_data(data):
    #pass





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
parser.add_argument('--no_color_fill', action="store_true", help='don\'t fill boxes with different colors')
parser.add_argument('--bar_ci', action="store_true", help='adds CIs to bar plots')
parser.add_argument('--bar_average', action="store_true", help='creates bar plots using averages')
parser.add_argument('--optix_src', action="store", help='Path to the optix benchmark dump which should be compared against')
parser.add_argument('--optix_add', action="store", nargs='+', help='Use multiple scene benchmarks provided here as a list instead of src to compare against optix')



args = parser.parse_args()

rays = []
ray_types = ["Primary", "AO", "Diffuse"]

benchmark_info, kernels, data = read_benchmark_file(args.src)


if args.plot_cnf is None:
    kernel_plot_metadata = {}
    output_kernels = kernels.copy()
else:
    k_list, kernel_plot_metadata = get_plot_cnf(args.plot_cnf)
    output_kernels = [x for x in k_list if x in kernels]

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

if args.optix_src is not None:
    # array of means/medians for each kernel 
    bar_data = []
    errors = []

    labels = []

    if args.optix_add is not None:

        plt.figure(figsize =(10, 7))
        fig, ax = plt.subplots(figsize = (10, 7))

        label_list = get_kernel_label_list(kernel_plot_metadata, output_kernels)
        label_list.append("optix")
        bar_data = [] 
        for idx in range(len(label_list)):
            bar_data.append([])

        print(bar_data)
        print("label list")
        print(label_list)
        plot_data = None
        for scene_benchmark in args.optix_add:

            benchmark_scene_info, scene_kernels, scene_data = read_benchmark_file(scene_benchmark)

            labels.append(benchmark_scene_info["obj_file"].split(".")[0])

            if args.plot_cnf is None:
                output_kernels = kernels.copy()
            raw_data = get_time_per_measurement_ms(benchmark_scene_info, scene_data, True)
            ao_p_data = get_primary_plus_ao_time(raw_data, scene_kernels, output_kernels)
            optix_measurements = get_optix_measurements(args.optix_src, benchmark_scene_info["obj_file"])

            num_mes = min(len(ao_p_data[0]), len(optix_measurements))
            print(optix_measurements)
            ao_p_data.append(optix_measurements)
            scene_kernels.append("optix")

            summarized_data = None
            if args.bar_average:
                summarized_data = get_arithmetic_means(ao_p_data)
            else:
                summarized_data = get_medians(ao_p_data)

            for idx, data_point in enumerate(summarized_data):
                d = bar_data[idx]
                d.append(data_point)
                print(bar_data)

        print("scene labels:")
        print(labels)

        x = np.arange(len(labels))

        num_items = len(args.optix_add)

        margin = 0.2
        width = (1.-2.*margin)/num_items

        for idx, kernel_label in enumerate(label_list):
            xdata = x+margin+(idx*width)
            print("kernel: %s" %(kernel_label))
            print(bar_data[idx])
            ax.bar(xdata, bar_data[idx], width, label=kernel_label)

        ax.set_xticks(x + 0.5)
        ax.set_xticklabels(labels)    

        #ax.legend()

        #barplot = plt.bar(label_list, plot_data)
        max_y_lim = max(summarized_data)*1.2
        min_y_lim = 0
        plt.ylim(min_y_lim, max_y_lim)
        
    

            

    
    #optix_measurements = get_optix_measurements(args.optix_src, benchmark_info["obj_file"])
    #ao_p_data = get_primary_plus_ao_time(mray_data, kernels, output_kernels)
    
    
    #label_list = get_kernel_label_list(kernel_plot_metadata, output_kernels)

    

    #boxprops = dict(linestyle='--', linewidth=1.5)
    #medianprops = dict(linewidth=1.5)
    #bplot = plt.boxplot(ao_p_data, notch=True, labels=label_list, patch_artist=True, boxprops=boxprops, medianprops=medianprops)

        

    if args.time:
        y_label = plt.ylabel('Runtime [ms]', fontsize=16, labelpad=10)
    else:
        y_label = plt.ylabel('Performance [Mrays / s]', fontsize=16, labelpad=10)

    #y_label.set_rotation(0)

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    #for idx, box in enumerate(bplot['boxes']):
    #        if is_reference_kernel(kernel_plot_metadata, output_kernels[idx]):
    #            box.set_facecolor('lightblue')
    #        else:
    #            box.set_facecolor('lightgreen')

    # set border
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)

    # set grid
    ax.yaxis.grid()
    ax.set_axisbelow(True)

    # format x-labels
    if args.xfmt:
        fig.autofmt_xdate()

    fig.tight_layout()

    plt.savefig(out_path + "optix_comp_" + str(benchmark_info["num_measure"]) + ".png")


    exit()

if args.box_plot or args.bar_plot:
    label_list = get_kernel_label_list(kernel_plot_metadata, output_kernels)
    for i in range(len(rays)):
        plot_data = get_plot_data(mray_data, i, kernels, output_kernels)
        #data = get_box_plot_data(mray_data, i)
        plt.figure(figsize =(10, 7))
        fig, ax = plt.subplots(figsize = (10, 7))

        #plt.figure()
        #fig, ax = plt.subplots()

        #plt.figure(figsize =(15, 10.5))
        #fig, ax = plt.subplots(figsize = (15, 10.5))
        # Creating plot
        
        bplot = None
        barplot = None

        

        if args.box_plot:
            boxprops = dict(linestyle='--', linewidth=1.5, facecolor = '#d9d9d9')
            medianprops = dict(linewidth=1.5)
            bplot = plt.boxplot(plot_data, notch=True, labels=label_list, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
        elif args.bar_plot:
            summarized_data = None
            y_error = None
            if args.bar_average:
                summarized_data = get_arithmetic_means(plot_data)
                y_error = mean_95ci_errors(plot_data)
                
            else:
                summarized_data = get_medians(plot_data)
                y_error = median_95ci_errors(plot_data)
                print(summarized_data)
                print(y_error)
                factor = statistics.NormalDist().inv_cdf((1+0.95)/2)
                print(factor)

            print(summarized_data)
            print(label_list)
            #low errors = [su]

            barplot = plt.bar(label_list, summarized_data, yerr=y_error, capsize=2, width=0.4)
            max_y_lim = max(summarized_data)*1.2
            min_y_lim = 0
            plt.ylim(min_y_lim, max_y_lim)

        

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
        if args.box_plot and not args.no_color_fill:
            for idx, box in enumerate(bplot['boxes']):
                if is_reference_kernel(kernel_plot_metadata, output_kernels[idx]):
                    box.set_facecolor('lightblue')
                else:
                    box.set_facecolor('lightgreen')
        elif args.bar_plot and not args.no_color_fill:
            for idx, bar in enumerate(barplot):
                bar.set_alpha(1.0)
                if is_reference_kernel(kernel_plot_metadata, output_kernels[idx]):
                    bar.set_facecolor('lightblue')
                else:
                    bar.set_facecolor('lightgreen')

        # set border
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)

        # set grid
        
        ax.yaxis.grid()
        ax.set_axisbelow(True)

        # format x-labels
        if args.xfmt:
            fig.autofmt_xdate()

        # add statistical data
        if args.plot_stat_data:
            text_props = dict(facecolor='wheat', edgecolor='black', boxstyle='round,pad=0.5')
            if args.bar_average:
                diff = mean_largest_diff_95ci(plot_data)
            else:
                diff = largest_diff_95ci(plot_data)

            ci_diff = "%.2f" % (diff*100)
            if args.bar_average:
                stat_str = r'$95 \% \ CI(mean): \pm$ '
            else:
                stat_str = r'$95 \% \ CI: \pm$ '

            stat_str += ci_diff + r'$\%$' 
            stat_str += "\n"
            stat_str += r'measurements: ' + str(benchmark_info["num_measure"])

            # place a text box in upper left in axes coords
            ax.text(0.05, 0.925, stat_str, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=text_props)  
                
        fig.tight_layout()
        
        suffix = ""
        if args.bar_plot:
            suffix += "_bar"
        if args.bar_average:
            suffix += "_mean"

        plt.savefig(out_path + scene_name + "_" + ray_types[i] + "_" + str(benchmark_info["num_measure"]) + suffix + "_" + args.plot_cnf.split('\\')[-1] + ".png")
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
    
            


