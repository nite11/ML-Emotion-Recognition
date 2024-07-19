import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import mediapipe as mp
import os
import csv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from datetime import datetime

# Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

def test_landmarker(image_file):
    json_data=[]
    image_data={}
    header=[]
    image = mp.Image.create_from_file(image_file)
    detection_result = detector.detect(image)
    #print(image_file)
    for face_blendshapes_category in detection_result.face_blendshapes[0]:
        image_data[face_blendshapes_category.category_name]=round(face_blendshapes_category.score,4)
        header.append(face_blendshapes_category.category_name)
    json_data.append(image_data)
    with open('testdata.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)  
        writer.writeheader()    
        writer.writerows(json_data)

data = pd.read_csv('trainingdata.csv')  
#print(data)

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=400, random_state=14)

model.fit(X_scaled, y)

directory='./test_images/'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_suffix = -1
for root, dirs, files in os.walk(directory):
    for filename in files:  
        file_suffix = file_suffix + 1        
        image_file = os.path.join(root, filename)
        img = Image.open(image_file).convert('RGB')
        img.save(image_file, 'JPEG')
        test_landmarker(image_file)        

        test_data = pd.read_csv('testdata.csv')  
        new_data = test_data.iloc[:, :]
        new_data_scaled = scaler.transform(new_data)
        predicted_category = model.predict(new_data_scaled)
        #img = Image.open(image_file)
        new_filename = f'{predicted_category[0]}_{timestamp}_{file_suffix}.jpg'
        new_filename = os.path.join(root, new_filename)
        #img.save(new_filename, 'PNG')
        os.rename(image_file, new_filename)