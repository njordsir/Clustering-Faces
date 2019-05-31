import numpy as np
#import cv2
import imageio
import imutils
import os
from glob import glob
from tqdm import tqdm
import pickle
import tensorflow as tf
from mtcnn import mtcnn
import warnings
warnings.filterwarnings('ignore')

"""load the networks for detecting faces
   https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf
   Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks"""
def load_networks():
    global sess, pnet, rnet, onet
    sess = tf.Session()
    pnet, rnet, onet = mtcnn.create_mtcnn(sess, 'mtcnn')

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44

"""detect faces and obtain crops
   append results to list"""
def get_crops(img, face_crops):
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, points = mtcnn.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) > 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                face_crops.append(cropped)

"""read image, resize and get faces with mtcnn"""
def get_faces(img_path, face_crops):
    #img = cv2.imread(img_path)
    img = imageio.imread(img_path)
    img = imutils.resize(img, width=1000)
    get_crops(img, face_crops)

"""process all images in the folder for faces
   returns all cropped faces"""
def process_folder(folder_path, use_cache=False, save_cache=False):
    print("Finding faces in " + folder_path)

    cache_path = os.path.join(folder_path, "face_crops.pkl")

    if use_cache:
        print("Loading cached faces...")
        if not os.path.exists(cache_path):
            print("No cache found in {0}!".format(cache_path))
            return []
        with open(cache_path, "rb") as f:
            face_crops_all = pickle.load(f)
    else:
        load_networks()
        
        face_crops_all = []
        for img_path in tqdm(glob(os.path.join(folder_path, "*jpg"))):
            get_faces(img_path, face_crops_all)
        sess.close()

        if save_cache:
            print("Saving cached faces...")
            with open(cache_path, "wb") as f:
                pickle.dump(face_crops_all, f)

    print("{0} faces found".format(len(face_crops_all)))
    return face_crops_all