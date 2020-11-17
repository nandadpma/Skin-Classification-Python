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
#d = ['90']
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
def ekstraksi_fitur(s_distance,t_distance):
    for distance in range(s_distance,t_distance):
        ekstraksi_GLCM(trainingDir,distance)
        ekstraksi_GLCM(testingDir,distance)
        GLCM.glcm.img_number = 0
def classification(s_distance,t_distance):
    for distance in range(s_distance,t_distance):
        dataTraining_filename, kelas_dataTraining = load_all_file(trainingDir,distance)
        dataTesting_filename, kelas_dataTesting = load_all_file(testingDir,distance)
        for every_dir in all_combination:
            for combination in every_dir:
                fit = ['-'.join(item) for item in combination]
                _dataTraining = GLCM.glcm.get_features(dataTraining_filename,fit)
                _dataTesting = GLCM.glcm.get_features(dataTesting_filename,fit)
                dataTraining = list(np.array(_dataTraining)[:,1:])
                dataTesting = list(np.array(_dataTesting)[:,1:])
                classifier = NuSVC(kernel='rbf', nu=0.120, gamma=0.070)
                #classifier = SVC(kernel='linear')
                svm_model = classifier.fit(dataTraining,kelas_dataTraining)
                result = svm_model.predict(dataTesting)
                kelas_hasil_prediksi = [ 'Berminyak' if item==1 else 'Tidak Berminyak' for item in result]
                kelas_sebenarnya = [ 'Berminyak' if item==1 else 'Tidak Berminyak' for item in kelas_dataTesting]
                akurasi = hitung_akurasi(result,kelas_dataTesting)
                hasil_klasifikasi = {'Data Uji':list(np.array(_dataTesting)[:,0]), 'Kelas Prediksi':kelas_hasil_prediksi, 'Kelas Sebenarnya':kelas_sebenarnya}
                df = pd.DataFrame(hasil_klasifikasi)
                df.to_excel('../Hasil Klasifikasi/[d-'+str(distance)+',Dir-'+str(combination[0,1])+']-hasil klasifikasi-['+str(akurasi)+'].xlsx')
                print('\n',df)
                print('Akurasi : ',akurasi,'% : d',str(distance),':',fit)
#ekstraksi_fitur(1,2)
classification(1,5)

