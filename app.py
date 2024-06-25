import cv2
from geti_sdk.deployment import Deployment
import numpy as np
from flask import Flask, request, jsonify
import base64
import requests
import shutil
import os
from urllib.parse import urlencode
import imghdr

app = Flask(__name__)

def decode_base64_image(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def get_image(id_survey, flag, types):
    base_url = "http://202.180.16.237:28088/api/minio/image"
    payload = {
        'surveyId': id_survey,
        'flag': flag,
        'type': types
    }
    # Encode the payload into a query string
    query_string = urlencode(payload)
    full_url = f"{base_url}?{query_string}"
    
    response = requests.get(full_url, headers={'Content-Type': 'application/json'})
    # Assuming the response content is the image
    if response.status_code == 200:
        return response.content
    else:
        response.raise_for_status()
    

@app.route('/api/detect_image', methods=['POST'])
def detect_images():
    deployment = Deployment.from_folder("/home/refky/websocket/Comvis Bappenas/Classification Service/deployment")
    deployment.load_inference_models(device="CPU")
    results = []
    data = []
    first_classId = None

    if request.method == 'POST':
        try:
            request_data = request.get_json()

            image_paths = []  # List untuk menyimpan path file gambar yang disimpan

            for i in range(1, 12):  # Mulai dari 1 hingga 11
                id_survey_key = f'id_survey{i}'
                flag_key = f'flag{i}'
                type_key = f'type{i}'

                id_survey = request_data.get(id_survey_key)
                flag = request_data.get(flag_key)
                type_ = request_data.get(type_key)

                if id_survey and flag and type_:
                    try:
                        image_content = get_image(id_survey, flag, type_)

                        save_folder = './saved_images'
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)

                        # Simpan image_content ke file JPG
                        file_path = os.path.join(save_folder, f'image_{id_survey}.jpg')
                        with open(file_path, 'wb') as image_file:
                            image_file.write(image_content)
                        
                        print(f"Image saved to: {file_path}")
                        image_paths.append(file_path)  # Tambahkan path gambar ke list

                    except Exception as e:
                        print(f"Error getting image {id_survey}: {e}")
                        results.append({
                            'id_survey': id_survey,
                            'error': str(e)
                        })
            

            class_count = {
                "1": 0,  # Pengairan
                "2": 0,  # Vegetatif 1
                "3": 0,  # Vegetatif 2
                "4": 0,  # Generatif 1
                "5": 0,  # Generatif 2
                "6": 0   # BERA
            }
            # Proses semua gambar yang sudah disimpan
            for image_path in image_paths:
                try:
                    img = cv2.imread(image_path)
                    # converting BGR to RGB 
                    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                    prediction = deployment.infer(image_rgb)
                    phase_detected = "Unknown"

                    for annot in prediction.annotations:
                        for lab in annot.labels:
                            scores = lab.probability
                            classId = lab.name
                            confidence = scores

                            if confidence > 0.7:
                                if classId in class_count:
                                    class_count[classId] += 1
                                    if classId == 1 or classId == "1":
                                        phase_detected = "Pengairan"
                                    elif classId == 2 or classId == "2":
                                        phase_detected = "Vegetatif 1"
                                    elif classId == 3 or classId == "3":
                                        phase_detected = "Vegetatif 2"
                                    elif classId == 4 or classId == "4":
                                        phase_detected = "Generatif 1"
                                    elif classId == 5 or classId == "5":
                                        phase_detected = "Generatif 2"
                                    elif classId == 6 or classId == "6":
                                        phase_detected = "BERA"


                                    sorted_classes = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
                                    
                                    first_classId = sorted_classes[0][0]
                    
                    surveyId = ''.join(filter(str.isdigit, image_path))

                    results.append({
                    'surveyId': surveyId,
                    'detectedPhase': phase_detected
                })
                    os.remove(image_path)
                    
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    return str(e)
                
            data.append({
                'mostDetected' : first_classId
            })

            return jsonify(data), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)