#import cv2
import imageio
import os
import shutil
from tqdm import tqdm
import pickle

"""create folders for each cluster and save photos"""
def create_results_folder(face_crops,
                          labels,
                          folder_path,
                          sffx="_res_"):
    results_base_path = os.path.join(folder_path, sffx)
    if os.path.exists(results_base_path):
        print("Deleting past results...")
        shutil.rmtree(results_base_path)
    
    with open("/home/shankar/Classes/all/face_names.pkl", "rb") as f:
        names = pickle.load(f)

    print("Saving new results...")
    for i in tqdm(range(len(face_crops))):
        cluster_idx = labels[i]
        crop = face_crops[i]
        cluster_folder_path = os.path.join(results_base_path, str(cluster_idx))
        if not os.path.exists(cluster_folder_path):
            os.makedirs(cluster_folder_path)
        
        #cv2.imwrite(os.path.join(cluster_folder_path, "_" + str(i) + ".jpg"), crop)
        imageio.imsave(os.path.join(cluster_folder_path, names[i]), crop, format=".jpg")