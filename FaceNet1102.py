# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 06:17:03 2021

@author: Ahmed Fayed
"""

from keras_facenet import FaceNet
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import Normalizer


embedder = FaceNet()
l2_normalizer = Normalizer('l2')


def read_image(path):
    
    image = cv2.imread(path)
    
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image_RGB



def detect(image):
    
    detections = embedder.extract(image, threshold=0.95)
    
    return detections


def get_points(box):
    
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    
    x2 = x1 + width
    y2 = y1 + height
    
    return (x1, y1), (x2, y2)


database = dict()

def add_person(image, name):
    
    detections = detect(image)
    
    database[name] = detections[0]['embedding']
    




# creating database

ahmed_img = read_image('E:/Software/Experiments/known/Ahmed1/Ahmed1.jpg') 
abdallah_img = read_image('E:/Software/Experiments/known/Abdallah/Abdallah.jpg')
dodo_img = read_image('E:/Software/Experiments/known/Dodo/Dodo.jpg')    
malek_img = read_image('E:/Software/Experiments/known/Malek/Malek.jpg')
rammah_img1 =read_image('E:/Software/Experiments/known/Rammah1/Rammah1.jpg')
rammah_img2 = read_image('E:/Software/Experiments/known/Rammah2/Rammah2.jpg')
rammah_img3 = read_image('E:/Software/Experiments/known/Rammah3/Rammah3.jpg')
rammah_img5 = read_image('E:/Software/Experiments/known/Rammah5/Rammah5.jpg')
rammah_img6 = read_image('E:/Software/Experiments/known/Rammah6/Rammah6.jpg')
rammah_img7 = read_image('E:/Software/Experiments/known/Rammah7/Rammah7.jpg')
rammah_img8 = read_image('E:/Software/Experiments/known/Rammah8/Rammah8.jpg')
sayed_img = read_image('E:/Software/Experiments/known/Sayed/Sayed.jpg')


add_person(ahmed_img, 'Ahmed')
add_person(abdallah_img, 'Abdallah')
add_person(dodo_img, 'Dodo')
add_person(malek_img, 'Malek')
add_person(sayed_img, 'Sayed')


# Multiple images for a single person experiment

# add_person(rammah_img1, 'Rammah1')
# add_person(rammah_img2, 'Rammah2')
# add_person(rammah_img3, 'Rammah3')


# rammah_img1 = cv2.resize(rammah_img1, (160, 160))
# rammah_img2 = cv2.resize(rammah_img2, (160, 160))
# rammah_img3 = cv2.resize(rammah_img3, (160, 160))


rammah1_encode = detect(rammah_img1)
rammah1_encode = rammah1_encode[0]['embedding']

rammah2_encode = detect(rammah_img2)
rammah2_encode = rammah2_encode[0]['embedding']

rammah3_encode = detect(rammah_img3)
rammah3_encode = rammah3_encode[0]['embedding']

rammah5_encode = detect(rammah_img5)
rammah5_encode = rammah5_encode[0]['embedding']

rammah6_encode = detect(rammah_img6)
rammah6_encode = rammah6_encode[0]['embedding']

rammah7_encode = detect(rammah_img7)
rammah7_encode = rammah7_encode[0]['embedding']

rammah8_encode = detect(rammah_img8)
rammah8_encode = rammah8_encode[0]['embedding']

rammah_encode_list = [rammah1_encode, rammah2_encode, rammah3_encode, rammah5_encode, rammah6_encode, rammah7_encode, rammah8_encode]
rammah_encode = np.sum(rammah_encode_list, axis=0)
rammah_encode = l2_normalizer.transform(np.expand_dims(rammah_encode, axis=0))[0]
database['Rammah'] = rammah_encode



# path = 'E:/Software/Experiments/IMG-20191208-WA0004.jpg'
# image = read_image(path)


# image_copy = image.copy()
# image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# detections = detect(image_copy)


counter = 0

for person_directory in os.listdir('E:/Software/Experiments/test/'):
    person_dir = os.path.join('E:/Software/Experiments/test/', person_directory)
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
    image = read_image(image_path)
    image_copy = image.copy()
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    
    # resizing experiment
    # image_copy = cv2.resize(image_copy, (2000, 1800))
    
    detections = detect(image_copy)
    
   

    for res in detections:
        
        pt1, pt2 = get_points(res['box'])
         
        min_distance = 100
        identity = "unknown"
        for person in database:
            
            embedding1 = database[person]
            embedding2 = res['embedding']
            distance = embedder.compute_distance(embedding1, embedding2)
            distance = np.round(distance, decimals=1)
            
            if distance < min_distance:
                min_distance = distance
                identity = person
                
                
        
        if min_distance <= 0.3:
            cv2.rectangle(image_copy, pt1, pt2, (0, 200, 200), 3)
            cv2.putText(image_copy, identity + f'_{min_distance:.1f}', (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.rectangle(image_copy, pt1, pt2, (0, 0, 255), 3)
            cv2.putText(image_copy, 'unknown', pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        print('identity:  ' + identity + ' ====>  ', min_distance)
     
         
    image_name_2 = 'test' + '{}.jpg'    
    cv2.imwrite(image_name_2.format(counter), image_copy)
    counter += 1
    # print(image.shape)
    image_copy = cv2.resize(image_copy, (1000, 900))
    # print(image_copy.shape)

    plt.imshow(image_copy)
    plt.show()
        
    
    
    

# cv2.imwrite('IMG-20191208-WA0004.jpg', image_copy)
# plt.imshow(image_copy)

# print(image.shape)
# image_copy = cv2.resize(image_copy, (1000, 900))
# print(image_copy.shape)


# while True:
    
#     cv2.imshow('result', image_copy)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cv2.destroyAllWindows()










