import cv2
from geti_sdk.deployment import Deployment
import numpy as np
from flask import Flask, request, jsonify
import base64

app = Flask(__name__)

def decode_base64_image(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

@app.route('/api/detect_image', methods=['POST'])
def detect_image():
    deployment = Deployment.from_folder("/home/refky/websocket/Comvis Bappenas/Classification Service/deployment")
    deployment.load_inference_models(device="CPU")
    res = ""
    if request.method == 'POST':
        try :
            request_data = request.get_json()

            image_base64 = request_data['image']
            image = decode_base64_image(image_base64)

            prediction = deployment.infer(image)

            for annot in prediction.annotations:
                for lab in annot.labels:
                    scores = lab.probability
                    classId = lab.name
                    confidence = scores

                    if confidence > 0.7 :
                        if classId == 1 or classId == "1" :
                            res = "Pengairan"
                        if classId == 2 or classId == "2" :
                            res = "Vegetatif 1"
                        if classId == 3 or classId == "3" :
                            res = "Vegetatif 2"
                        if classId == 4 or classId == "4" :
                            res = "Generatif 1"
                        if classId == 5 or classId == "5" :
                            res = "Generatif 2"
                        if classId == 6 or classId == "6" :
                            res = "BERA"

            return jsonify({'result': classId}), 200
            
        except Exception as e :
            return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")