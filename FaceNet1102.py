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




embedder = FaceNet()


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
rammah_img = read_image('E:/Software/Experiments/known/Rammah3/Rammah3.jpg')
sayed_img = read_image('E:/Software/Experiments/known/Sayed/Sayed.jpg')


add_person(ahmed_img, 'Ahmed')
add_person(abdallah_img, 'Abdallah')
add_person(dodo_img, 'Dodo')
add_person(malek_img, 'Malek')
add_person(rammah_img, 'Rammah')
add_person(sayed_img, 'Sayed')

# path = 'E:/Software/Experiments/IMG-20191208-WA0004.jpg'
# image = read_image(path)


# image_copy = image.copy()
# image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# detections = detect(image_copy)


counter = 0

for person_directory in os.listdir('E:/Software/Experiments/test_0/'):
    person_dir = os.path.join('E:/Software/Experiments/test_0/', person_directory)
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
    image = read_image(image_path)
    image_copy = image.copy()
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    detections = detect(image_copy)

    for res in detections:
        
        pt1, pt2 = get_points(res['box'])
        
        min_distance = 100
        identity = "unknown"
        for person in database:
            
            embedding1 = database[person]
            embedding2 = res['embedding']
            distance = embedder.compute_distance(embedding1, embedding2)
            
            if distance < min_distance:
                min_distance = distance
                identity = person
                
                
        
        if min_distance < 0.5:
            cv2.rectangle(image_copy, pt1, pt2, (0, 200, 200), 3)
            cv2.putText(image_copy, identity + f'_{min_distance:.2f}', (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
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










