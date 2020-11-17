import numpy as np
import cv2
from skimage import io, util
from skimage.feature.texture import greycomatrix, greycoprops
from PIL import Image

img = io.imread('..\contoh_fullsize.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('..\gray', gray)
#cv2.imshow("h",gray)
#cv2.waitKey()
cv2.imwrite('..\gray.png', gray)
gray = np.array(gray,dtype=np.uint8)
#print(np.shape(gray))
#print(type(gray))
#glcm = greycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
#glcm = greycomatrix(gray, [1], [0], levels=256, normed=True, symmetric=True)
#glcm = greycomatrix(gray, [1], [0, 3*np.pi/4, np.pi/2, np.pi/4], levels=256)
d = 1
glcm = greycomatrix(gray, [d], [0, 3*np.pi/4, np.pi/2, np.pi/4], levels=256)
print(np.shape(glcm))

print(glcm[0:7,0:7,0,0])

diss = greycoprops(glcm, 'dissimilarity')
contrast = greycoprops(glcm, 'contrast')
correlation = greycoprops(glcm, 'correlation')
homogeneity = greycoprops(glcm, 'homogeneity')
ASM = greycoprops(glcm, 'ASM')
print(homogeneity[0,0],' ',homogeneity[0,1],' ',homogeneity[0,2],' ',homogeneity[0,3])

def calc_glcm(gray, index_baris, index_kolom, x_offset, y_offset):
        x, y = gray.shape
        temp_glcm = np.zeros((256,256))
        for i in index_baris:
            for j in index_kolom:
                current = gray[i, j]
                neighbor = gray[i + x_offset, j + y_offset]
                temp_glcm[current,neighbor]+=1
        sim = temp_glcm + temp_glcm.transpose()
        return np.array([np.array([item/np.sum(sim) for item in baris]) for baris in sim])
def Homogeneity(glcm):
        x, y = glcm.shape
        return np.array([[glcm[i, j]/(1+np.power(np.abs(i-j),2)) for j in range(y)] for i in range(x)]).sum()
def ASM(glcm):
        return np.power(glcm,2).sum()
def Contrast(glcm):
        x, y = glcm.shape
        return np.array([[np.power(i-j,2)* glcm[i, j] for j in range(y)] for i in range(x)]).sum()
def miu_y(x, y, glcm):
        return float(np.array([[j*glcm[i, j] for j in range(y)] for i in range(x)]).sum())
def miu_x(x, y, glcm):
        return float(np.array([[i*glcm[i, j] for j in range(y)] for i in range(x)]).sum())
def std_dev_y(x, y, glcm, miu_y):
        return float(np.array([[np.power((1 - miu_y), 2)*glcm[i, j] for j in range(y)] for i in range(x)]).sum())
def std_dev_x(x, y, glcm, miu_x):
        return float(np.array([[np.power((1 - miu_x), 2)*glcm[i, j] for j in range(y)] for i in range(x)]).sum())
def Correlation(glcm):
        x, y = glcm.shape
        temp_miu_x = miu_x(x, y, glcm)
        temp_miu_y = miu_y(x, y, glcm)
        temp_std_dev_x = std_dev_x(x, y, glcm, temp_miu_x)
        temp_std_dev_y = std_dev_y(x, y, glcm, temp_miu_y)
        return np.array([[((1-temp_miu_x)*(1-temp_miu_y)*glcm[i,j])/(temp_std_dev_x*temp_std_dev_y) for j in range(y)] for i in range(x)]).sum()

'''
distance = 1
x,y = np.shape(gray)
glcm0 = calc_glcm(gray, range(0,x), range(0,y-distance), 0, distance)
glcm45 = calc_glcm(gray, range(distance,x), range(0,y-distance), -distance, distance)
glcm90 = calc_glcm(gray, range(distance,x), range(0,y), -distance, 0)
glcm135 = calc_glcm(gray, range(distance,x), range(distance,y), -distance, -distance)
print(Homogeneity(glcm0),' ',Homogeneity(glcm45),' ',Homogeneity(glcm90),' ',Homogeneity(glcm135))
'''
#print(glcm)
'''
rows, cols, bands = img.shape

radius = 5
side = 2*radius + 1

distances = [1]
angles = [0, np.pi/2]
props = ['contrast', 'dissimilarity', 'homogeneity']
dim = len(distances)*len(angles)*len(props)*bands

padded = np.pad(img, radius, mode='reflect')
windows = [util.view_as_windows(padded[:, :, band].copy(), (side, side))
           for band in range(bands)]
feats = np.zeros(shape=(rows, cols, dim))

for row in range(rows):
    for col in range(cols):
        pixel_feats = []
        for band in range(bands):
            glcm = greycomatrix(windows[band][row, col, :, :],
                                distances=distances,
                                angles=angles)
            pixel_feats.extend([greycoprops(glcm, prop).ravel()
                                for prop in props])
        feats[row, col, :] = np.concatenate(pixel_feats)
'''
'''
image = np.array(img, dtype=np.uint8)
glcm = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=4)
#greycoprops(glcm, 'dissimilarity')
#greycoprops(glcm, 'correlation')
#print(a,b)
print(glcm)
print('finish')
'''
