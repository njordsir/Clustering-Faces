import importlib
from collections import Counter
import os
import sys
import argparse
import pickle
import numpy as np

from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
if importlib.util.find_spec("MulticoreTSNE") is None:
    from sklearn.manifold import TSNE
else:
    from MulticoreTSNE import MulticoreTSNE as TSNE

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

dlib_exists = importlib.util.find_spec("dlib") is not None
if dlib_exists:
    from dlib import chinese_whispers_clustering
    from dlib import vector as dlib_vector

from utils import create_results_folder

def get_tsne(face_encodings):
    """get the tsne embeddings of the face encodings for visualizations"""
    tsne_output = TSNE(n_jobs=-1).fit_transform(face_encodings)
    return tsne_output

def viz_tsne(face_crops, face_encodings, labels = []):
    """plot tsne embeddings with label colors"""
    tsne_output = get_tsne(face_encodings)
    
    colors = None
    if len(labels) > 0:
        if len(set(labels)) <= 13:
            colors_set = [ "red", "blue", "green", "yellow", "purple", "orange", 
                       "black", "gray", "magenta", "cyan", "pink", "chartreuse", "white"]
            colors = [colors_set[x] for x in list(labels)]
        else:
            colors_set = np.random.uniform(low=0.0, high=1.0, size=len(set(labels)))
            colors = [colors_set[x] for x in list(labels)]

    visualize_clusters(tsne_output[:,0], tsne_output[:,1], face_crops, colors=colors)

#reference : https://stackoverflow.com/questions/42867400/python-show-image-upon-hovering-over-a-point
def visualize_clusters(x, y, imgs, colors=None):
    """create an interactive plot visualizing the clusters
       hovering over a point shows the corresponding face crop"""
    # create figure and plot scatter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    if colors:
        line, = ax.plot(x,y, ls="")
        ax.scatter(x,y,c=colors)
    else:
        line, = ax.plot(x,y, ls="", marker="o")

    # create the annotations box
    im = OffsetImage(imgs[0], zoom=1)
    xybox=(50., 50.)
    ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            # find out the index within the array from the event
            ind, = line.contains(event)[1]["ind"]
            # get the figure size
            w,h = fig.get_size_inches()*fig.dpi
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0]*ws, xybox[1]*hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy =(x[ind], y[ind])
            # set the image corresponding to that point
            #im.set_data(imgs[ind][:,:,::-1])
            im.set_data(imgs[ind])
        else:
            #if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)           
    plt.show()

cluster_methods = ["kmeans", "dbscan", "chinese_whispers"]

def cluster(encodings,
            method=cluster_methods[2],
            k=2, #kmeans
            eps=17, min_samples=30, #dbscan
            cw_min_delta=20, #chinese_whispers
            prune=False, prune_min_samples=10 #prune small clusters
            ):
    if method == cluster_methods[0]:
        print("Performing KMeans clustering with k =", k)
        clusterModel = KMeans(n_clusters=k, max_iter=5000, n_init=25, n_jobs=-1)
        clusterModel.fit(encodings)
        labels = clusterModel.labels_

    elif method == cluster_methods[1]:
        print("Performing DBScan clustering with eps =", eps)
        clusterModel = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        clusterModel.fit(encodings)
        labels = clusterModel.labels_

    elif method == cluster_methods[2]:
        if dlib_exists:
            print("Performing Chinese Whispers clustering with eps =", cw_min_delta)
            encodings_dlib = [dlib_vector([float(x) for x in list(row)]) 
                                           for row in encodings]
            labels = chinese_whispers_clustering(encodings_dlib, cw_min_delta)
        else:
            print("Chinese Whispers clustering requires dlib!")
            return None

    else:
        print("Invalid cluster method!")
        return

    if prune:
        clusters_to_drop = []
        cnt_labels = Counter(labels)
        for c in cnt_labels:
            if cnt_labels[c] < prune_min_samples:
                clusters_to_drop.append(c)

        for i,label in enumerate(labels):
            if label in clusters_to_drop:
                labels[i] = -1

        print("Dropped {0} clusters of size less than {1}"
                .format(len(clusters_to_drop), prune_min_samples))

    n_identified = len(set(labels))
    print("Number of people identified = {0}".format(n_identified))
    
    """show the distribution of clusters"""
    dist = dict(Counter(labels))
    dist = [dist[k] for k in sorted(dist, key=lambda x:dist[x], reverse=True)]
    #print("Cluster distribution :", dist)
    
    return labels

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f","--folder", type=str,
        help='folder containing photos to be processed',
        required=True)

    parser.add_argument(
        "-c","--cluster_method", type=str,
        help='one of ["kmeans", "dbscan", "chinese_whispers"]',
        default='chinese_whispers',
        required=True)

    parser.add_argument(
        "-k","--kmeans_k", type=int,
        help='number of clusters for k-means clustering',
        default=2,
        required=False)

    parser.add_argument(
        "-eps","--dbscan_eps", type=float,
        help='eps value for DBScan clustering',
        default=17,
        required=False)
    parser.add_argument(
        "-ms","--dbscan_min_samples", type=int,
        help='minimum sample count for a cluster in DBScan clustering',
        default=5,
        required=False)
    parser.add_argument(
        "-cw","--chinese_whispers_eps", type=float,
        help='eps value for ChineseWhispers clustering',
        default=20,
        required=False)

    parser.add_argument(
        "-p","--prune", type=bool,
        help='ignore clusters with size less than threshold',
        default=False, required=False)
    parser.add_argument(
        "-pms","--prune_threshold", type=int,
        help='threshold for pruning smaller clusters',
        default=30,
        required=False)

    parser.add_argument(
        "-v","--visualize", type=bool,
        help='see interactive vizualizations for clusters',
        default=False, required=False)

    parser.add_argument(
        "-s","--save", type=bool,
        help='save cluster results to folders',
        default=False, required=False)

    return parser.parse_args(argv)

def main(args):
    crops_path = os.path.join(args.folder, "face_crops.pkl")
    enc_path = os.path.join(args.folder, "face_encodings.pkl")

    if not os.path.exists(crops_path):
        print("ERROR : Face crops not found! Generate them before clustering.")
        return
    if not os.path.exists(enc_path):
        print("ERROR : Face encodings not found! Generate them before clustering.")
        return

    with open(crops_path, "rb") as f:
        face_crops = pickle.load(f)
    with open(enc_path, "rb") as f:
        face_encodings = pickle.load(f)

    if args.cluster_method not in cluster_methods:
        print('ERROR : Clustering method should be one of ["kmeans", "dbscan", "chinese_whispers"]')
        return

    labels = cluster(encodings=face_encodings,
                     method=args.cluster_method,
                     k=args.kmeans_k,
                     eps=args.dbscan_eps, min_samples=args.dbscan_min_samples,
                     cw_min_delta=args.chinese_whispers_eps,
                     prune=args.prune, prune_min_samples=args.prune_threshold
                    )

    if args.save:
        create_results_folder(face_crops, labels, args.folder)

    if args.visualize:
        viz_tsne(face_crops, face_encodings, labels)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))