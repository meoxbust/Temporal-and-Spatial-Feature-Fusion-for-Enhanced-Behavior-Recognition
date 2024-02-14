import os
import cv2
import numpy as np


def read_frames(array_videos, each_nth=15):
    videos=[]
    n = len(array_videos)
    for i in range(n):
        print(np.round(100*i/n, 3))
        frames = []
        cnt = 0
        num_imgs = len(os.listdir(array_videos[i]))
#         print(num_imgs)
        for image in os.listdir(array_videos[i]):
#             print(cnt)
            if cnt == (num_imgs // 2):
#                 print('append')
                image = image.decode('utf-8')
                img = cv2.imread(os.path.join(array_videos[i], image))
# #             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (128, 128))
#             frames.append(img)
                videos.append(img)
            cnt += 1
    return np.array(videos)