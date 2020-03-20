import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', '-d', default='dataset/test')
parser.add_argument('--file_name', '-f', default='test.csv')
parser.add_argument('--is_train', type=bool, default=False)

args = parser.parse_args()

if __name__ == '__main__':
    train_df: pd.DataFrame = pd.read_csv(os.path.join(args.data_dir, args.file_name))

    num_row, num_col = train_df.shape

    for index in tqdm(range(num_row)):
        image: str = train_df.iloc[index]['Image']
        image: np.ndarray = np.fromstring(image, dtype=int, sep=' ')
        image = image.reshape((96, 96))
        if not args.is_train:
            index = train_df.iloc[index]['ImageId']
        cv2.imwrite(os.path.join(args.data_dir, f'{index}.png'), image)

    train_df: pd.DataFrame = train_df.drop('Image', 1)
    train_df.to_csv(os.path.join(args.data_dir, args.file_name), index=False)