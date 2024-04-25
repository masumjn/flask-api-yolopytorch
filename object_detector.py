from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image
from flask_cors import CORS
import os
from random2 import randint
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

uploads_dir = os.path.join(app.instance_path, 'upload')
output_dir = os.path.join(app.instance_path, 'results')

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
    # with open("indexurl.html") as file:
        return file.read()


@app.route("/detect/", methods=["POST"])
def detect():
    """
        Handler of /detect POST endpoint
        Receives uploaded file with a name "image_file", passes it
        through YOLOv8 object detection network and returns and array
        of bounding boxes.
        :return: a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    try:
        os.mkdir(uploads_dir)
        os.mkdir(output_dir)
    except:
        pass
    buf = request.files["image_file"]
    filename = "img_source"
    target_filename = secure_filename(filename + '.jpg')
    target_path = os.path.join(uploads_dir, target_filename)

    if os.path.exists(target_path):
        os.remove(target_path)  # Remove the existing file

    buf.save(target_path)  # Save the new file

    boxes = detect_objects_on_image(buf.stream)
    return jsonify(boxes)



def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO("best.pt")
    results = model.predict(Image.open(buf))
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output


# serve(app, host='0.0.0.0', port=8080)
# serve(app, host='0.0.0.0', port=5000)
if __name__ == '__main__':
    app.run(debug=True)
