import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import shutil
from typing import List, Tuple

def download_and_save_images(data: List[dict], image_save_path: str, image_size: Tuple[int, int] = (250, 250)) -> pd.DataFrame:
    """
    Downloads images from the given URLs and saves them to a specified directory.

    Args:
        data (List[dict]): List of dictionaries containing image URLs and SKU information.
        image_save_path (str): Path to the directory where the images will be saved.
        image_size (Tuple[int, int], optional): Tuple of (width, height) for resizing the images.
                                               Default is (250, 250).

    Returns:
        pd.DataFrame: DataFrame containing SKU and corresponding image names.
    """
    info = []
    for element in data:
        response = requests.get(element['image'])
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image_name = str(element['sku']) + '.jpg'
            info.append([element['sku'], image_name])
            image = image.convert('RGB')
            image = image.resize(image_size)
            image.save(image_save_path + image_name)
    info_df = pd.DataFrame(info, columns=['SKU', 'ImageName'])
    return info_df

def copy_images(t: List[Tuple[str, str]], source_dir: str, destination_dir: str) -> None:
    """
    Copies images from the source directory to the destination directory based on the information in 't'.

    Args:
        t (List[Tuple[str, str]]): List of tuples containing SKU and corresponding folder name.
        source_dir (str): Path to the source directory where the images are located.
        destination_dir (str): Path to the destination directory where images will be copied.

    Returns:
        None
    """
    for element in t:
        source_image = source_dir + str(element[0]) + '.jpg'
        destination_image = destination_dir + '/' + element[1] + '/' + str(element[0]) + '.jpg'
        shutil.copy(source_image, destination_image)

