import re
import GLCM
import glob
import cv2
from scipy import stats
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import sklearn.preprocessing as scaling
import numpy as np
import matplotlib.pyplot as plt
import os.path
from itertools import combinations, product
import pandas as pd
f = ['ASM','Homogeneity','Contrast','Correlation']
d = ['0','45','90','135']
fc = list(list(combinations(f, i)) for i in [4])   
dc = list(list(combinations(d, i)) for i in [1])
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
testingDir = '../DataUji/*.jpg'
#testingDir = '../DataLatih/*.jpg'
def hitung_akurasi(result,actual):
    F = T = 0
    for i in range(len(result)):
        if(actual[i]==result[i]): T+=1
        else: F+=1
    return round(T/(T+F)*100,1)
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
def ekstraksi_GLCM(directory,distance):
    for img in glob.glob(directory):
        filename = re.search(pattern,img).group(0)
        image = cv2.imread(img)
        GLCM.glcm(image,filename,distance)
def ekstraksi_ciri(s_distance,t_distance):
    for distance in range(s_distance,t_distance):
        ekstraksi_GLCM(trainingDir,distance)
        ekstraksi_GLCM(testingDir,distance)
        GLCM.glcm.img_number = 0
def classification(distance):
    all_distance = []
    for d in distance:
        result_number = 0
        dataTraining_filename, kelas_dataTraining = load_all_file(trainingDir,d)
        dataTesting_filename, kelas_dataTesting = load_all_file(testingDir,d)
        all_dir = []
        indeks = [ ', '.join(list(item[:,0])) for item in all_combination[0]]
        for every_dir in all_combination:
            one_dir = []
            for combination in every_dir:
                fit = ['-'.join(item) for item in combination]
                result_number += 1
                dataTraining = GLCM.glcm.get_features(dataTraining_filename,fit)
                dataTesting = GLCM.glcm.get_features(dataTesting_filename,fit)
                all_data = (list(np.array(dataTraining)[:,1:])+list(np.array(dataTesting)[:,1:]))
                dataTraining = list(np.array(dataTraining)[:,1:])
                dataTesting = list(np.array(dataTesting)[:,1:])
                classifier = NuSVC(kernel='rbf', nu=0.120, gamma=0.070)
                #classifier = SVC(kernel='linear')
                svm_model = classifier.fit(dataTraining,kelas_dataTraining)
                result = svm_model.predict(dataTesting)
                akurasi = hitung_akurasi(result,kelas_dataTesting)
                print(akurasi,'% :',result_number,':D',str(d),':',fit)
                all_dir.append(akurasi)
        all_distance.append(all_dir)
    best = [ np.max(row) for row in all_distance]
    mean = [ np.mean(row) for row in all_distance] 
    #result = [[ str(item)+'%' for item in row]for row in np.transpose(all_distance)]
    result = [[ str(item).replace('.',',')+'%' for item in row]for row in np.transpose(all_distance)]
    all_test = {'0':result[0], '45':result[1], '90':result[2], '135':result[3], 'Max':best, 'Mean':mean}
    df = pd.DataFrame(all_test, index =distance)
    df.to_excel('../Hasil Klasifikasi/rangkuman_all_test.xlsx')
    print(df)
#ekstraksi_ciri(1,2)
classification(range(1,11))

