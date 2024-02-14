import numpy as np

def select_rows(videos_skeletons, n=15):
    videos = []
    for i in range(len(videos_skeletons)):
        frames = []
        video_size = len(videos_skeletons[i])
        for t in np.linspace(0, video_size - 1, num=n):
            frames.append(videos_skeletons[i][int(t)])
        videos.append(frames)

    videos = np.array(videos)
    print(videos.shape)
    return videos