import numpy as np
import pandas as pd

x_columns = [f'PointX_{i}' for i in range(1, 14)]
y_columns = [f'PointY_{i}' for i in range(1, 14)]
def read_csv_data(array_csv, each_nth=15):
    skeletons_videos = []
    n = len(array_csv)
    for i in range(n):
        print(np.round(100*i/n, 3))
        frames = []
        cnt = 0
        csv_reader = pd.read_csv(array_csv[i])
        print(array_csv[i], csv_reader.shape[0])
        for j in range(0, csv_reader.shape[0], 1):
                sequence = []
                for z in range(13):
                    point_x = csv_reader[x_columns[z]][j]
                    point_y = csv_reader[y_columns[z]][j]
                    sequence.extend([point_x, point_y])
                frames.append(sequence)
        skeletons_videos.append(frames)
    return skeletons_videos