import base64
import zlib

import cv2
import numpy as np

import os
from PIL import Image

import pandas as pd

from tqdm import tqdm


def encode(img: np.ndarray) -> bytes:
    """
    Lossless encoding of images for submission on kaggle platform.

    Parameters
    ----------
    img : np.ndarray
        cv2.imread(f) - BGR image in (h, w, c) format, c = 3 (even for png format).

    Returns
    -------
    bytes
        Encoded image as bytes.
    """
    img_to_encode = img.astype(np.uint8)
    img_to_encode = img_to_encode.flatten()
    img_to_encode = np.append(img_to_encode, -1)

    cnt, rle = 1, []
    for i in range(1, img_to_encode.shape[0]):
        if img_to_encode[i] == img_to_encode[i - 1]:
            cnt += 1
            if cnt > 255:
                rle += [img_to_encode[i - 1], 255]
                cnt = 1
        else:
            rle += [img_to_encode[i - 1], cnt]
            cnt = 1

    compressed = zlib.compress(bytes(rle), zlib.Z_BEST_COMPRESSION)
    base64_bytes = base64.b64encode(compressed)
    return base64_bytes


def prediction(
    sample_submission: str,
    lr_folder: str,
    output_file: str
) -> None:
    """
    Get prediction for sample submission using huggingface model.

    Parameters
    ----------
    sample_submission : str
        Path to sample submission file.
    lr_folder : str
        Path to test dataset folder with LR images.
    output_file : str
        Path to output file with predictions.
    model : str
        Name (repository) of huggingface model.
    device : str
        Device on which the model will be run.
    simple_resize : str
            If specified then the upscaling will be done using deterministic interpolation.

    Returns
    -------
    None
    """
    submission_df = pd.read_csv(sample_submission)

    filenames = submission_df["filename"].values
    for i in tqdm(range(len(filenames))):
        filename = filenames[i]
        filename = filename.split(".")[0] + "_DRCT-L_X4.png"
        init_img = cv2.imread(os.path.join(lr_folder, filename))
        submission_df.loc[i, "rle"] = encode(init_img)
    submission_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    prediction(
        sample_submission="./dataset/sample_submission.csv",
        lr_folder='./kaggle_test_HR_result/pre-trained_with_kaggle_dataset_with_DACLIP_TTA_v3_95000',
        output_file=f'kaggle_solution.csv'
    )