import mediapipe as mp
import os
import json
import csv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
#from visualization import draw_landmarks_on_image
#from PIL import Image as im

# Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

directory='./training_data/'

json_data=[]
header=[]

for root, dirs, files in os.walk(directory):
    for filename in files:  
        image_file = os.path.join(root, filename)
        emotion = filename[:2].lower()
        if emotion == 'co':
                continue
        image_data={}
        image_data['emotion'] = emotion
        header=[]
        header.append('emotion')
# Load the input image.
        image = mp.Image.create_from_file(image_file)

# Detect face landmarks from the input image.
        detection_result = detector.detect(image)

# visualize the detection result
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# data=im.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# data.save('./landmarked_img.png') 
        
        for face_blendshapes_category in detection_result.face_blendshapes[0]:
                image_data[face_blendshapes_category.category_name]=round(face_blendshapes_category.score,4)
                header.append(face_blendshapes_category.category_name)
        json_data.append(image_data)
# print the transformation matrix.
# print(detection_result.facial_transformation_matrixes)
with open('trainingdata.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    writer.writerows(json_data)

# with open("features.json", "w") as features:
#     json.dump(header[1:], features, indent=4)