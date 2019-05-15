import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

main_dir = r'/home/alexander/Downloads/P21'
base_name = 'P212'

new_file_name = r'/home/alexander/Downloads'+ base_name + '.csv'

plot_figures = False
manual = False

list_dir = os.listdir(main_dir)
list_dir.sort()

def peak_finder(array, width=1):
    peak_indices = []
    for i in range(width, len(array) - width):
        max_value = np.max(array[i-width:i+width])
        if array[i] == max_value:
            peak_indices.append(i)
    return np.array(peak_indices)


def data_cleaner(data):
    data = data[91:217, 1]
    for i, value in enumerate(data):
        if value is None:
            data[i] = (data[i-1]+data[i+1])/2.
    return data

def data_detrender(data):
    peaks = peak_finder(-data, width=9)
    if len(peaks)>1:
        X = np.vstack([peaks, np.ones(peaks.shape)]).T
        m, c = np.linalg.lstsq(X, data[peaks])[0]
        data_detrend = data - (c + m * np.linspace(0, len(data) - 1, len(data)))
    else:
        data_detrend = data - data[peaks]

    return data_detrend, peaks

def get_output_value(data_detrend):
    w = 1.
    window = 5
    index_180 = 130 - 91 + np.argmax(data_detrend[130 - 91:134 - 91])
    index_190 = 138 - 91 + np.argmax(data_detrend[138 - 91:143 - 91])
    index_147 = 104 - 91 + np.argmax(data_detrend[104 - 91:108 - 91])
    index_256 = 190 - 91 + np.argmax(data_detrend[190 - 91:203 - 91])


    I_180 = w*np.sum(data_detrend[index_180-window:index_180+window])
    I_190 = w*np.sum(data_detrend[index_190-window:index_190+window])
    I_147 = w*np.sum(data_detrend[index_147-window:index_147+window])
    I_256 = w*np.sum(data_detrend[index_256-window:index_256+window])

    output_value = float(I_180 + I_190) / float(
        I_180 + I_190 + 0.32 * (I_147 + I_256))

    return output_value

plt.figure()
outputs = np.array([])
for file_name in list_dir:
    print('processing file {} ...'.format(file_name))
    file_path = main_dir + '/' + file_name
    data = np.loadtxt(file_path)
    data = data_cleaner(data)
    data_detrend, peaks = data_detrender(data)
    if plot_figures:
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(data)
        plt.plot(peaks, data[peaks], '*')
        plt.subplot(2,1,2)
        plt.plot(data_detrend)
        plt.plot(peaks, data_detrend[peaks], '*')
        plt.show()

    # I_180 = np.max(data_detrend[130-91:134-91])
    # I_190 = np.max(data_detrend[138-91:143-91])
    # I_147 = np.max(data_detrend[104-91:108-91])
    # I_256 = np.max(data_detrend[190-91:203-91])
    #
    # output_value = float(I_180 + I_190)/float(I_180 + I_190 + 0.32*(I_147 + I_256))
    output_value = get_output_value(data_detrend)
    outputs = np.append(outputs, output_value)
    print('output_value: {}'.format(output_value))
    if manual:
        input('press enter to continue')

output_file = pd.DataFrame(outputs)
output_file.to_csv(new_file_name)



# file_name = list_dir[0]
# file_path = main_dir + '/' + file_name
# data = np.loadtxt(file_path)
# data = data_cleaner(data)
# peaks = peak_finder(-data, width = 5)
# plt.plot(data)
# plt.plot(peaks,data[peaks],'*')
# plt.show()
# X = np.vstack([peaks, np.ones(peaks.shape)]).T
# m, c = np.linalg.lstsq(X, data[peaks])[0]
#
# data_detrend = data - (c+m*np.linspace(0,len(data)-1,len(data)))
# plt.figure()
# plt.plot(data_detrend)
# plt.plot(peaks,data_detrend[peaks],'*')
# plt.show()

