from mxnet import recordio
import mxnet as mx
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
# path_imgidx = '/home/q519111/face/ms1m-retinaface-t1/train.idx'
# path_imgrec = '/home/q519111/face/ms1m-retinaface-t1/train.rec'
path_imgidx = '/home/q519111/face/faces_emore/train.idx'
path_imgrec = '/home/q519111/face/faces_emore/train.rec'

imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

# i = 0
pids = set()
for i in tqdm(range(2000)):
    header, s = recordio.unpack(imgrec.read_idx(i+1))
    print(header.label)
    # pid = int(header.label[0])
    pid = int(header.label)
    pids.add(pid)
    # img = mx.image.imdecode(s).asnumpy()
    # pid_folder = os.path.join("/home/q519111/Pictures/face",str(pid))
    # os.makedirs(pid_folder,exist_ok=True)
    # out_path = os.path.join(pid_folder,str(i)+".jpg")
    # cv2.imwrite(out_path,img)
    # print(pid)
    # print(img.shape)

print("Total ids:",len(pids))