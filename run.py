import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Function to perform template matching
def match_template(main_image, template):
    result = cv2.matchTemplate(main_image, template, cv2.TM_CCORR_NORMED)
   
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

    return (minVal, maxVal, minLoc, maxLoc)


def process_image(image_path, template_path):
    try:
        main_image = cv2.imread(image_path)

        if main_image is None:
            raise Exception(f"Failed to read the main image at {image_path}")

        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        if template is None:
            raise Exception(f"Failed to read the template image at {template_path}")

        cumulative_startX = 0
        cumulative_startY = 0
        cumulative_endX = 0
        cumulative_endY = 0
        num_matches = 0

        for scale in [0.8, 0.9, 1.0, 1.2]:
            scaled_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            (minVal, maxVal, minLoc, maxLoc) = match_template(
                main_gray, scaled_template
            )

            threshold = 0.9
            print("maxval: ", maxVal)
            if maxVal > threshold:
                if not is_identical_match(maxLoc):
                    continue

                (startX, startY) = maxLoc
                endX = startX + int(scaled_template.shape[1] / scale)
                endY = startY + int(scaled_template.shape[0] / scale)

                cumulative_startX += startX
                cumulative_startY += startY
                cumulative_endX += endX
                cumulative_endY += endY

                num_matches += 1

        if num_matches > 0:
            avg_startX = cumulative_startX // num_matches
            avg_startY = cumulative_startY // num_matches
            avg_endX = cumulative_endX // num_matches
            avg_endY = cumulative_endY // num_matches

            # Draw rectangle on the image
            cv2.rectangle(
                main_image,
                (avg_startX, avg_startY),
                (avg_endX, avg_endY),
                (255, 0, 0),
                3,
            )

            # Crop the selected area

            # cropped_area = main_image[avg_startY:avg_endY, avg_startX:avg_endX]
            cropped_area = main_image
            # Apply noise cancellation (denoising)
            denoised_cropped_area = cv2.fastNlMeansDenoisingColored(cropped_area, None, 10, 10, 7, 21)

            # Use YOLO for object detection
            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            classes = []
            with open("coco.names", "r") as f:
                classes = f.read().strip().split("\n")

            layer_names = net.getUnconnectedOutLayersNames()

            blob = cv2.dnn.blobFromImage(
                denoised_cropped_area, 0.00392, (416, 416), (0, 0, 0), True, crop=False
            )
            net.setInput(blob)
            outs = net.forward(layer_names)

            # Extract information about detected objects
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * cropped_area.shape[1])
                        center_y = int(detection[1] * cropped_area.shape[0])
                        w = int(detection[2] * cropped_area.shape[1])
                        h = int(detection[3] * cropped_area.shape[0])
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            # Draw rectangles around detected objects
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    cv2.rectangle(
                        denoised_cropped_area,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2,
                    )


            # Save the denoised and object-detected cropped area as the modified image
            uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
            modified_images_dir = os.path.join(uploads_dir, "modified_images")
            os.makedirs(modified_images_dir, exist_ok=True)
            modified_image_path = os.path.join(
                modified_images_dir,
                f"modified_" + os.path.basename(image_path),
            )
            cv2.imwrite(
                modified_image_path, denoised_cropped_area, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )

            _, img_encoded = cv2.imencode(".jpg", denoised_cropped_area)
            modified_image_base64 = base64.b64encode(img_encoded).decode("utf-8")

            return {
                "message": "Template match found",
                "modified_image_path": modified_image_path,
                "modified_image_base64": modified_image_base64,
            }

        return {"message": "No template match found"}

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def is_identical_match(current_loc):
    # Check if the current match template location is too close to the previous one
    # Adjust the threshold as needed
    threshold_distance = 20
    if 'prev_loc' not in globals() or np.linalg.norm(np.array(current_loc) - np.array(prev_loc)) > threshold_distance:
        globals()['prev_loc'] = current_loc
        return True
    return False

@app.route('/process', methods=['POST'])
def process_uploaded_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Ensure the 'uploads' directory exists
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        template_dir = os.path.join(os.path.dirname(__file__), 'model')
        os.makedirs(template_dir, exist_ok=True)
        # Save the uploaded file
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        # Define the path to the template image
        template_path = os.path.join(template_dir, 'model.jpg')

        # Process the image and check for template match
        processed_result = process_image(file_path, template_path)

        return jsonify(processed_result)

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)