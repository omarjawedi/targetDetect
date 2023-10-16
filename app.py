import os
from flask import Flask, request, jsonify
import cv2
import numpy as np


app = Flask(__name__)

def match_template(main_image, template):
    result = cv2.matchTemplate(main_image, template, cv2.TM_CCORR_NORMED)
   
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

    return (minVal, maxVal, minLoc, maxLoc)

def process_image(image_path, template_path):
    try:
        main_image = cv2.imread(image_path)

        if main_image is None:
            raise Exception(f"Failed to read the main image at {image_path}")
        
        main_template = cv2.imread(template_path)

        main_image = cv2.fastNlMeansDenoisingColored(main_image, None, 10, 10, 7, 21)
        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)


        template = cv2.cvtColor(main_template, cv2.COLOR_BGR2GRAY)



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

            cv2.rectangle(
                main_image,
                (avg_startX, avg_startY),
                (avg_endX, avg_endY),
                (255, 0, 0),
                3,
            )

            uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
            modified_images_dir = os.path.join(uploads_dir, "modified_images")
            os.makedirs(modified_images_dir, exist_ok=True)
            modified_image_path = os.path.join(
                modified_images_dir,
                f"modified_" + os.path.basename(image_path),
            )
            cv2.imwrite(
                modified_image_path, main_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )

            return {
                "message": "Template match found",
                "modified_image_path": modified_image_path
            }

        return {"message": "No template match found"}

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def is_identical_match(current_loc):

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

        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        template_dir = os.path.join(os.path.dirname(__file__), 'model')
        os.makedirs(template_dir, exist_ok=True)

        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        template_path = os.path.join(template_dir, 'modeltest.jpg')

        processed_result = process_image(file_path, template_path)

        return jsonify(processed_result)

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)

