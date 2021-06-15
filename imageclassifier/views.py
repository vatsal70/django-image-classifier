from django.http import HttpResponse
from django.shortcuts import render, redirect

def index(request):
    return render(request, 'index.html')



# Machine Learning Project
import joblib
import json
import numpy as np
import base64
import cv2
from .wavelet import w2d
from matplotlib import pyplot as plt
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage


__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_base64_data, file_path=None):

    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    # print(imgs)
    result = []
    try:
        for img in imgs:
            scalled_raw_img = cv2.resize(img, (32, 32))
            img_har = w2d(img, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (32, 32))
            combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
            len_image_array = 32*32*3 + 32*32

            final = combined_img.reshape(1,len_image_array).astype(float)
            load_saved_artifacts()
            result.append({
                'class': class_number_to_name(__model.predict(final)[0]),
                'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
                'class_dictionary': __class_name_to_number,
                'number': __model.predict(final)[0],
            })
            # print(result)
        return result
    except:
        print("Something went wrong.")



def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    with open("C:/Users/vatsal70/Desktop/imageclassifier/imageclassifier/artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('C:/Users/vatsal70/Desktop/imageclassifier/imageclassifier/artifacts/saved_model_logistic_regression.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('C:/Users/vatsal70/Desktop/imageclassifier/imageclassifier/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:/Users/vatsal70/Desktop/imageclassifier/imageclassifier/opencv/haarcascades/haarcascade_eye.xml')
    try:
        if image_path:
            img = cv2.imread(image_path)
        else:
            img = get_cv2_image_from_base64_string(image_base64_data)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        cropped_faces = []
        for (x,y,w,h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) >= 2:
                    cropped_faces.append(roi_color)
        return cropped_faces
    except:
        print("something went wrong")




def class_image(request):
    if request.method == 'POST':
        # image_data = request.FILES['image_data']
        image_d = request.FILES.get('image_data')
        print(image_d)
        fs = FileSystemStorage()
        filename = fs.save(image_d.name, image_d)
        uploaded_file_url = fs.url(filename)
        main_url = "C:/Users/vatsal70/Desktop/imageclassifier/"+ uploaded_file_url
        response = classify_image(None, str(image_d))
        print(response)
        params = {
            'response': response,
        }
        return render(request, 'classify_image.html', params)
    return render(request, 'classify_image.html')