import json, numpy as np, tensorflow as tf, cv2 as cv, dlib
from typing import List
from os.path import join
from glob import glob
from imageio import imread, imsave
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

INPUT_SIZE = 256

# Initialize
app = FastAPI()

tf.reset_default_graph()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph(join('models', 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0') # source
Y = graph.get_tensor_by_name('Y:0') # reference
Xs = graph.get_tensor_by_name('generator/xs:0') # output

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

def preprocess(img):
    return img.astype(np.float32) / 127.5 - 1.
    return (img / 255. - 0.5) * 2

def postprocess(img):
    return ((img + 1.) * 127.5).astype(np.uint8)
    return (img + 1) / 2

def align_faces(img):
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = shape_predictor(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
    return faces

@app.post('/predict')
async def predict(in_files: List[UploadFile] = File(...)):
    
    with open('result.jpg', 'wb+') as file_object:
        file_object.write(in_files[0].file.read())

    no_makeup = cv.resize(imread('result.jpg'), (INPUT_SIZE, INPUT_SIZE))
    X_img = np.expand_dims(preprocess(no_makeup), 0)
    makeups = glob(join('imgs', 'makeup', '*.*'))
    result = np.ones((2 * INPUT_SIZE, (len(makeups) + 1) * INPUT_SIZE, 3))
    result[INPUT_SIZE: 2 *  INPUT_SIZE, :INPUT_SIZE] = no_makeup / 255.

    for i in range(len(makeups)):
        makeup = cv.resize(imread(makeups[i]), (INPUT_SIZE, INPUT_SIZE))
        Y_img = np.expand_dims(preprocess(makeup), 0)
        Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        Xs_ = postprocess(Xs_)
        result[:INPUT_SIZE, (i + 1) * INPUT_SIZE: (i + 2) * INPUT_SIZE] = makeup / 255.
        result[INPUT_SIZE: 2 * INPUT_SIZE, (i + 1) * INPUT_SIZE: (i + 2) * INPUT_SIZE] = Xs_[0]
    imsave('result.jpg', result)
    return FileResponse('result.jpg')
