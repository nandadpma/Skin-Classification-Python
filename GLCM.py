import numpy as np
import cv2
import pymongo
class glcm():
    connection = pymongo.MongoClient('localhost',27017)
    db_features = connection['SkinClf']
    collection = db_features['GLCM_FEATURES']
    db_result = connection['SkinClsf']
    result_collection = db_result['CLS_RESULT_dt']
    img_number = 0
    def __init__(self, file, filename, distance):
        self.gray = self.RGB2Gray(file)
        self.filename = filename
        self.distance  = distance
        self.hitungMatrixGLCM()
    def check_duplicated(self,doc_id):
        match_doc = glcm.collection.find({'_id':doc_id})
        duplicated_count = match_doc.count()
        return duplicated_count
    def insert_one_Document(self,doc_id,data):
        glcm.collection.insert_one(data)
    def check_duplicated_result(doc_id):
        match_doc = glcm.result_collection.find({'_id':doc_id})
        duplicated_count = match_doc.count()
        glcm.connection.close()
        return duplicated_count
    def insert_result(data):
        glcm.result_collection.insert_one(data)
        glcm.connection.close()
    def RGB2Gray(self, image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    def calc_glcm(self, gray, index_baris, index_kolom, x_offset, y_offset):
        x, y = gray.shape
        temp_glcm = np.zeros((256,256))
        for i in index_baris:
            for j in index_kolom:
                current = gray[i, j]
                neighbor = gray[i + x_offset, j + y_offset]
                temp_glcm[current,neighbor]+=1
        sim = temp_glcm + temp_glcm.transpose()
        return np.array([np.array([item/np.sum(sim) for item in baris]) for baris in sim])
    def calculate_glcm(self):
        x, y = self.gray.shape
        self.glcm0 = self.calc_glcm(self.gray, range(0,x), range(0,y-self.distance), 0, self.distance)
        self.glcm45 = self.calc_glcm(self.gray, range(self.distance,x), range(0,y-self.distance), -self.distance, self.distance)
        self.glcm90 = self.calc_glcm(self.gray, range(self.distance,x), range(0,y), -self.distance, 0)
        self.glcm135 = self.calc_glcm(self.gray, range(self.distance,x), range(self.distance,y), -self.distance, -self.distance)
    def hitungMatrixGLCM(self):
        self.doc_id = 'D'+str(self.distance)+'_'+self.filename
        if self.check_duplicated(self.doc_id) == 0:
            self.calculate_glcm()
            data = {'_id': self.doc_id,'ASM-0':self.ASM(self.glcm0),'Contrast-0':self.Contrast(self.glcm0),'Homogeneity-0':self.Homogeneity(self.glcm0),'Correlation-0':self.Correlation(self.glcm0)
            ,'ASM-45':self.ASM(self.glcm45),'Contrast-45':self.Contrast(self.glcm45),'Homogeneity-45':self.Homogeneity(self.glcm45),'Correlation-45':self.Correlation(self.glcm45)
            ,'ASM-90':self.ASM(self.glcm90),'Contrast-90':self.Contrast(self.glcm90),'Homogeneity-90':self.Homogeneity(self.glcm90),'Correlation-90':self.Correlation(self.glcm90)
            ,'ASM-135':self.ASM(self.glcm135),'Contrast-135':self.Contrast(self.glcm135),'Homogeneity-135':self.Homogeneity(self.glcm135),'Correlation-135':self.Correlation(self.glcm135)}
            self.insert_one_Document(self.doc_id,data)
            glcm.img_number = glcm.img_number + 1
            print(glcm.img_number,' : ',self.doc_id)
    def get_features(docs_id, docs_features):
        result = [list(i.values())for i in glcm.collection.find({'_id':{'$in':docs_id}},docs_features)]
        glcm.connection.close()
        return result
    def get_Results(n_combination,docs_id):
        result = [list(i.values())for i in glcm.result_collection.find({'_id':{'$in':docs_id}})]
        glcm.connection.close()
        return result
    def miu_y(self, x, y, glcm):
        return float(np.array([[j*glcm[i, j] for j in range(y)] for i in range(x)]).sum())
    def miu_x(self, x, y, glcm):
        return float(np.array([[i*glcm[i, j] for j in range(y)] for i in range(x)]).sum())
    def std_dev_y(self, x, y, glcm, miu_y):
        return float(np.array([[np.power((1 - miu_y), 2)*glcm[i, j] for j in range(y)] for i in range(x)]).sum())
    def std_dev_x(self, x, y, glcm, miu_x):
        return float(np.array([[np.power((1 - miu_x), 2)*glcm[i, j] for j in range(y)] for i in range(x)]).sum())
    def Correlation(self, glcm):
        x, y = glcm.shape
        temp_miu_x = self.miu_x(x, y, glcm)
        temp_miu_y = self.miu_y(x, y, glcm)
        temp_std_dev_x = self.std_dev_x(x, y, glcm, temp_miu_x)
        temp_std_dev_y = self.std_dev_y(x, y, glcm, temp_miu_y)
        return np.array([[((1-temp_miu_x)*(1-temp_miu_y)*glcm[i,j])/(temp_std_dev_x*temp_std_dev_y) for j in range(y)] for i in range(x)]).sum()
    def Homogeneity(self, glcm):
        x, y = glcm.shape
        return np.array([[glcm[i, j]/(1+np.abs(i-j)) for j in range(y)] for i in range(x)]).sum()
    def Contrast(self, glcm):
        x, y = glcm.shape
        return np.array([[np.power(i-j,2)* glcm[i, j] for j in range(y)] for i in range(x)]).sum()
    def ASM(self, glcm):
        return np.power(glcm,2).sum()