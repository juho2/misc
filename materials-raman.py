import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import xlrd
#from collections import defaultdict

def read_all_xlsx(path):
    all_xls = [os.path.join(root, name) for root, dirs, files in os.walk(path) 
    for name in files if name.endswith(('.xlsx', '.xls'))]
    all_data = {}
    for file in all_xls:
        label = file.split('\\')[-1].split('.')[0]
        book = xlrd.open_workbook(file)
        sheet = book.sheet_by_index(0)
        try:
            data = np.array([sheet.col_values(0), sheet.col_values(1)])
        except:
            data = sheet.col_values(0)
        all_data[label] = data
    return(all_data)

def pre_process(data, start=400):
    # single spectrum input
    # handle single-column
    if len(data) > 2:
#        print('Single column')
        temp_data = []
        for row in data:
            try:
                temp_data.append([float(row.split()[0]), float(row.split()[1])])
            except:
#                print('Discarded: ' + str(row))
                continue
        data = np.array(temp_data).T
    # remove single-line header
    try:
        data = data.astype(np.float)
    except:
#        print('Discarded non-numeric: ' + str(data[:,0]))
        data = data[:,1:]
        data = data.astype(np.float)

#    # discard pump, start at max between start = [a,b]
#    start_0 = (np.abs(data[0] - start[0])).argmin()
#    start_1 = (np.abs(data[0] - start[1])).argmin()
#    data = data[:, start_0:]
#    startpoint = data[1,:start_1-start_0].argmax()
#    data = data[:, startpoint:]
    # start at fixed value
    data = data[:, start:]
        
    # baseline correction - should fit to valleys instead !
    def func(x, a, b, c, d):
        return(a*x**3 + b*x**2 + c*x + d)
        #return(a*np.exp(-b*x)+c)
    popt, pcov = curve_fit(func, data[0], data[1], [1, 1, 1, 1])
    fit = func(data[0], *popt)
    data[1] = data[1]-fit

    # normalize to [0,1]
    data[1] -= min(data[1])
    data[1] /= np.max(data[1])
    # interpolate to 3000 data points
    f = interp1d(data[0], data[1])
    x = np.linspace(min(data[0]), max(data[0]), 3000)
    y = f(x)
    data = np.array([x,y])
    return(data)
    
def plot_data(data, title=''):
    # data type: dict of spectra
    nrows = 4
    ncols = int(np.ceil(len(data)/nrows))
    fig = plt.figure(title)
    count = 1
    for label in sorted(data):
        ax = fig.add_subplot(nrows, ncols, count)
        ax.plot(data[label][0], data[label][1])
        plt.legend([label])
#        print(label, count)
        count += 1

def identify(pp_test_data, pp_pure_data, pure_th=0.3):
    
#    plt.figure('test data')
#    plt.plot(pp_test_data[0], pp_test_data[1])
    errors_pure, residuals = {}, {}
    for label in pp_pure_data:
        residuals[label] = pp_test_data[1] - pp_pure_data[label][1]
        errors_pure[label] = np.sum(residuals[label]**2) # least squares
    min_error = min(errors_pure.values())
    best = list(errors_pure.keys())[list(errors_pure.values()).index(min_error)]
    if errors_pure[best] < pure_th:
        print('Pure ' + best + ' (error ' + str(errors_pure[best]) + ')\n')
        is_mixture = False
    else:
        print('Mixture, main component: ' + best + ' (error ' + str(errors_pure[best]) + ')')
        is_mixture = True        
#    plt.figure()
#    plt.plot(residuals[best][0], residuals[best][1])
#    print(errors_pure)
    for label in residuals:
        residuals[label] = np.array([pp_test_data[0], residuals[label]])
    return(best, residuals, is_mixture)

def identify_mixture(main_comp, residual, pp_pure_data):
    # normalize residual
    residual[1] -= min(residual[1])
    residual[1] /= np.max(residual[1])
    errors_pure, residuals, convos = {}, {}, {}
    for label in pp_pure_data:
        residuals[label] = residual[1] - pp_pure_data[label][1]
        errors_pure[label] = np.sum(residuals[label]**2) # least squares
        convos[label] = np.convolve(pp_pure_data[main_comp][1], pp_pure_data[label][1], mode='same')
        convos[label] /= np.max(convos[label])
    min_error = min(errors_pure.values())
    best = list(errors_pure.keys())[list(errors_pure.values()).index(min_error)]
    print(errors_pure)
    print('Secondary component: ' + best + ' (error ' + str(errors_pure[best]) + ')\n')
    for label in residuals:
        residuals[label] = np.array([residual[0], residuals[label]])
        convos[label] = np.array([np.arange(len(convos[label])), convos[label]])
    return(best, residuals, convos)

data_path = 'raman'
pure_data = read_all_xlsx(os.path.join(data_path, 'pure'))
test_data = read_all_xlsx(os.path.join(data_path, 'test'))
#plot_data(pure_data)

pp_pure_data = {}
FFTs = {}
for label in pure_data:
    pp_pure_data[label] = pre_process(pure_data[label])
    FFTs[label] = np.zeros(pp_pure_data[label].shape)
    freq = np.fft.fftfreq(pp_pure_data[label].shape[-1])
    Y = np.fft.fft(pp_pure_data[label][1])
#    Y -= min(Y)
    Y /= np.max(Y)
    FFTs[label] = np.vstack((freq, Y))
plot_data(pp_pure_data, 'Pre-processed pure spectra')
plot_data(FFTs, 'FFTs, pure spectra')

main_res, sec_res, mainres_FFTs, sec_convos = {}, {}, {}, {}
for label in test_data:
    print('Testing file: ' + label)
    pp_test_data = pre_process(test_data[label])
    main_comp, main_res[label], is_mixture = identify(pp_test_data, pp_pure_data)
    if is_mixture:
        sec_comp, sec_res[label], sec_convos[label] = identify_mixture(main_comp, main_res[label][main_comp], pp_pure_data)
#        plot_data(sec_res[label], label + ', secondary component')
    mainres_FFTs[label] = main_res[label]
    for pure in main_res[label]:
        freq = np.fft.fftfreq(main_res[label][pure].shape[-1])
        Y = np.fft.fft(main_res[label][pure][1])
        Y = -Y        
#        Y -= min(Y)
        Y /= np.max(Y)
        mainres_FFTs[label][pure] = np.vstack((freq, Y))
#    plot_data(main_res[label], label + ' main_res')
#    plot_data(sec_res[label], label + ' sec_res')
    plot_data(FFTs, 'fft')
    plot_data(mainres_FFTs[label], 'fft')
    plot_data(sec_convos[label], 'convos')
    
#mixture_path = path + r'\mixture\\'
#mixture_data = {}
#for folder in os.listdir(mixture_path):
#    print(mixture_path + folder)
#    mixture_data[folder] = read_all_xlsx(mixture_path + folder + r'\\')
#    
#pp_mixture_data = {}
#for folder in mixture_data:
#    pp_mixture_data[folder] = {}
#    for label in mixture_data[folder]:
##        print(label)
#        pp_mixture_data[folder][label] = pre_process(mixture_data[folder][label])
#    plot_data(pp_mixture_data[folder])


### sweep findpeaks with different params, get "meas error" from 10x mix scan, remove peaks and analyze residual

#detect_peaks(test[1], mpd = 200, valley=True, show=True)
#scikit-spectra

# x = pure_data['Motor Oil'][1]
# from detect_peaks import detect_peaks
# ind = detect_peaks(x, show=True)
# print(ind)
# 
