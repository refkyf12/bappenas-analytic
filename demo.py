# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import cv2
from geti_sdk.deployment import Deployment
from geti_sdk.utils import show_image_with_annotation_scene

if __name__ == "__main__":
    # Step 1: Load the deployment
    deployment = Deployment.from_folder("/home/refky/websocket/Comvis Bappenas/Classification Service/deployment")

    # Step 2: Load the sample image
    image = cv2.imread("/home/refky/websocket/Comvis Bappenas/Classification Service/E_PT1_E_20240320_154421.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 3: Send inference model(s) to CPU
    deployment.load_inference_models(device="CPU")

    # Step 4: Infer image
    prediction = deployment.infer(image_rgb)
    for annot in prediction.annotations:
        for lab in annot.labels:
            scores = lab.probability
            classId = lab.name
            confidence = scores

            if confidence > 0.7 :
                print(classId)