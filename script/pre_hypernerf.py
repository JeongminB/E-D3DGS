import os
import numpy as np
import glob
import sys
import json
from PIL import Image
from tqdm import tqdm
import shutil
import argparse
import sqlite3

sys.path.append(".")
from scripts.thirdparty.helper3dg import getcolmapsinglehyper


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def preparecolmap_hypernerf(path): 
    projectfolder = os.path.join(path, "colmap")
    manualfolder = os.path.join(projectfolder, "manual")

    # colmap_dir = os.path.join(root_dir,"sparse_")
    if not os.path.exists(manualfolder):
        os.makedirs(manualfolder)

    imagecolmap_dir = os.path.join(projectfolder,"input")
    if not os.path.exists(imagecolmap_dir):
        os.makedirs(imagecolmap_dir)

    image_dir = os.path.join(path,"rgb","2x")
    images = os.listdir(image_dir)
    images.sort()

    camera_dir = os.path.join(path,"camera")
    cameras = os.listdir(camera_dir)
    cameras.sort()

    cams = []
    for jsonfile in tqdm(cameras):
        with open (os.path.join(camera_dir,jsonfile)) as f:
            cams.append(json.load(f))
    
    image_size = cams[0]['image_size']
    image = Image.open(os.path.join(image_dir,images[0]))

    object_images_file = open(os.path.join(manualfolder,"images.txt"),"w")
    object_cameras_file = open(os.path.join(manualfolder,"cameras.txt"),"w")
    object_point_file = open(os.path.join(manualfolder,"points3D.txt"),"w")

    idx=0
    cnt=0
    sizes=2
    while len(cams)//sizes > 200:
        sizes += 1

    for cam, image in zip(cams, images):
        cnt+=1
        if cnt %  sizes != 0:
            continue

        R = np.array(cam['orientation']).T
        T = -np.array(cam['position'])@R 
        T = [str(i) for i in T]

        qevc = [str(i) for i in rotmat2qvec(R.T)]
        print(idx+1," ".join(qevc)," ".join(T),1,image,"\n",file=object_images_file)

        print(idx,"SIMPLE_PINHOLE",image_size[0]/2,image_size[1]/2,cam['focal_length']/2,cam['principal_point'][0]/2,cam['principal_point'][1]/2,file=object_cameras_file)
        idx+=1
        shutil.copy(os.path.join(image_dir,image),os.path.join(imagecolmap_dir,image))

    print(idx)

    object_cameras_file.close()
    object_images_file.close()
    object_point_file.close()

def converthypernerftocolmapdb(videopath):
    database_path = os.path.join(videopath, 'colmap/input.db')
    cameras_file = os.path.join(videopath, 'colmap/manual/cameras.txt')
    images_file = os.path.join(videopath, 'colmap/manual/images.txt')

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS cameras')
    cursor.execute('DROP TABLE IF EXISTS images')

    cursor.execute('''
    CREATE TABLE cameras (
        camera_id INTEGER PRIMARY KEY,
        model INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        params BLOB NOT NULL,
        prior_focal_length INTEGER NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE images (
        image_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        camera_id INTEGER NOT NULL,
        prior_qw REAL NOT NULL,
        prior_qx REAL NOT NULL,
        prior_qy REAL NOT NULL,
        prior_qz REAL NOT NULL,
        prior_tx REAL NOT NULL,
        prior_ty REAL NOT NULL,
        prior_tz REAL NOT NULL
    )
    ''')

    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}


    with open(cameras_file, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            camera_id = int(data[0]) 
            model = camModelDict[data[1]]
            width = int(float(data[2]))
            height = int(float(data[3]))
            params = np.array(data[4:], dtype=np.float64).tobytes()
            prior_focal_length = 1
            cursor.execute('''
            INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (camera_id, model, width, height, params, prior_focal_length))

    with open(images_file, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            if not data:
                continue
            image_id = int(data[0])  # idx + 1
            prior_qw = float(data[1])  # qw
            prior_qx = float(data[2])  # qx
            prior_qy = float(data[3])  # qy
            prior_qz = float(data[4])  # qz
            prior_tx = float(data[5])  # tx
            prior_ty = float(data[6])  # ty
            prior_tz = float(data[7])  # tz
            camera_id = int(data[8])  # camera ID
            name = data[9]  # image file name
            cursor.execute('''
            INSERT INTO images (image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz))


    conn.commit()
    conn.close()




if __name__ == "__main__" :
    parser = argparse.ArgumentParser() 
    parser.add_argument("--videopath", default="", type=str)

    args = parser.parse_args()
    videopath = args.videopath

    # # # ## step 1 prepare colmap input 
    print("start preparing colmap image input")
    preparecolmap_hypernerf(videopath)


    # # # ## step 2 prepare colmap db file 
    print("start preparing colmap database input")
    converthypernerftocolmapdb(videopath)

    ## step 3 run colmap, if error, reinstall opencv-headless 
    getcolmapsinglehyper(videopath)


