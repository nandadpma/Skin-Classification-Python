import re
import GLCM
import glob
import numpy as np
import os.path
from itertools import combinations, product
import pandas as pd
f = ['ASM','Homogeneity','Contrast','Correlation']
d = ['0','45','90','135']
fc = list(list(combinations(f, i)) for i in [4])   
dc = list(list(combinations(d, i)) for i in [4])
all_combination = np.array([ np.array(comb) for comb in ([ np.array(list(product(i, j))) for x in fc for i in x] for y in dc for  j in y)])
berminyak = 'Minyak'
pattern = '(\d{2,3}_\w*)'
dataTraining = []
kelas_dataTraining = []
dataTraining_filename = []
dataTesting = []
kelas_dataTesting = []
dataTesting_filename = []
trainingDir = '../DataLatih/*.jpg'
#testingDir = '../DataUji/*.jpg'
testingDir = '../show data/data uji/minyak/*.jpg'
#testingDir = '../show data/data uji/tidak/*.jpg'
#testingDir = '../show data/data latih/minyak/*.jpg'
#testingDir = '../DataLatih/*.jpg'
def load_all_file(directory,distance):
    data_filename = []
    kelas_data = []
    for img in glob.glob(directory):
        if berminyak in img: kelas = 1
        else: kelas = -1
        filename = re.search(pattern,img).group(0)
        data_filename.append('D'+str(distance)+'_'+filename)
        kelas_data.append(kelas)
    return data_filename,kelas_data
def show_data(s_distance,t_distance):
    all_distance = []
    for distance in range(s_distance, t_distance):
        result_number = 0
        dataTraining_filename, kelas_dataTraining = load_all_file(trainingDir,distance)
        dataTesting_filename, kelas_dataTesting = load_all_file(testingDir,distance)
        for every_dir in all_combination:
            for combination in every_dir:
                fit = ['-'.join(item) for item in combination]
                result_number += 1
                dataTraining = GLCM.glcm.get_features(dataTraining_filename,fit)
                dataTesting = GLCM.glcm.get_features(dataTesting_filename,fit)
                all_data = (list(np.array(dataTraining)[:,1:])+list(np.array(dataTesting)[:,1:]))
                #dataTraining = list(np.array(dataTraining)[:,1:])
                #dataTesting = list(np.array(dataTesting)[:,1:])
                fname = np.array(dataTesting)[:,0]
                data = np.array([[ round(float(item),10) for item in row] for row in np.array(dataTesting)[:,1:]])
                #print(dataTesting[:,0])
                #all_test = {'0':result[0], '45':result[1], '90':result[2], '135':result[3], 'Semua Arah':result[4]}
                all_data = {'ASM-0':data[:,0], 'Contrast-0':data[:,1], 'Homogeneity-0':data[:,2], 'Correlation-0':data[:,3],
                'ASM-45':data[:,4], 'Contrast-45':data[:,5], 'Homogeneity-45':data[:,6], 'Correlation-45':data[:,7],
                'ASM-90':data[:,8], 'Contrast-90':data[:,9], 'Homogeneity-90':data[:,10], 'Correlation-90':data[:,11],
                'ASM-135':data[:,12], 'Contrast-135':data[:,13], 'Homogeneity-135':data[:,14], 'Correlation-135':data[:,15]}
                df = pd.DataFrame(all_data, index =fname)
                print(df)
                #print(type(data[0,0]))
                df.to_excel('data latih berminyak'+str(distance)+'.xlsx')
        #print(df)
show_data(2,3)

