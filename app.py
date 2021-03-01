import os
import time
import threading
from flask import Response, Flask
import tensorflow as tf
import numpy as np
import cv2
import pickle
from imutils.video import VideoStream
from PIL import Image

import facenet
import detect_face
lock = threading.Lock()

output_face_frame = None
output_food_frame = None

app = Flask(__name__)

# Initiating video streams
face_stream = VideoStream(src=2).start()
food_stream = VideoStream(src=0).start()
time.sleep(2.0)


def core_detector():
    global face_stream, food_stream, output_face_frame, output_food_frame
    sensitivity = 0.75
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './npy')

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps threshold
            factor = 0.709  # scale factor
            image_size = 182
            input_image_size = 160

            facenet.load_model('./weights/20170511-185253.pb')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser('./weights/classifier.pkl')
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            # Load Food Detection Model
            net = cv2.dnn.readNet("weights/pedge-lite-model.weights", "cfg/iv-model.cfg")
            classes = []
            with open("cfg/obj.names", "r") as f:
                classes = [line.strip() for line in f.readlines()]

            with open("cfg/sub.names", "r") as f:
                person_ids = [line.strip() for line in f.readlines()]

            person_ids = person_ids
            person_ids.sort()

            students = person_ids

            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            while True:
                face_frame = face_stream.read()
                
                if face_frame is None:
                    print("Empty Face Frame")
                    time.sleep(0.1)
                    continue

                frame = cv2.resize(face_frame, (0, 0), fx=0.5, fy=0.5)

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet,
                                                            threshold, factor)
                nrof_faces = bounding_boxes.shape[0]

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('Face is very close!')
                            break

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(np.array(Image.fromarray(cropped[i]).resize(size=(
                            image_size, image_size))))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)),
                                                               best_class_indices]

                        if best_class_probabilities > sensitivity:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            for H_i in person_ids:
                                if person_ids[best_class_indices[0]] == H_i:
                                    result_names = person_ids[best_class_indices[0]]
                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                
                food_frame = food_stream.read()
                
                if food_frame is None:
                    print("Empty Food Frame")
                    time.sleep(0.1)
                    continue

                height, width, channels = food_frame.shape

                # Detecting objects
                blob = cv2.dnn.blobFromImage(food_frame, 0.00392, (416, 416), (0, 0, 0), True,
                                             crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)
                font = cv2.FONT_HERSHEY_PLAIN
                colors = np.random.uniform(0, 255, size=(len(classes), 3))
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.2:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        color = colors[class_ids[i]]
                        cv2.rectangle(food_frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(food_frame, label, (x, y + 30), font, 3, color, 3)

                with lock:
                    output_face_frame = frame.copy()
                    output_food_frame = food_frame.copy()


def generate_face():
    global output_face_frame, lock
    while True:
        with lock:
            if output_face_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_face_frame)

            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


def generate_food():
    global output_food_frame, lock
    while True:
        with lock:
            if output_food_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_food_frame)

            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/face_video_feed")
def face_video_feed():
    return Response(generate_face(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/food_video_feed")
def food_video_feed():
    return Response(generate_food(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    t = threading.Thread(target=core_detector)
    t.daemon = True
    t.start()

    app.run('0.0.0.0')
