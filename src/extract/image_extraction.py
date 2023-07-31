import argparse
import json
from extract import download_and_save_images, copy_images

def main(json_file: str, source_dir: str, destination_dir: str) -> None:
    """
    Main function to download images from JSON data and copy them to a specified directory.

    Args:
        json_file (str): Path to the JSON file containing image URLs and SKU information.
        source_dir (str): Path to the source directory where the images are located.
        destination_dir (str): Path to the destination directory where images will be copied.

    Returns:
        None
    """
    # Load JSON data from the file
    with open(json_file) as file:
        data_anyone = json.load(file)

    # Download and save images, then get SKU and image name information
    info_df = download_and_save_images(data_anyone)

    # Copy images to the destination directory
    image_name_df = info_df[['SKU', 'ImageName']].itertuples(index=False)
    copy_images(list(image_name_df), source_dir, destination_dir)

    # Save SKU and image name information to a CSV file
    info_df.to_csv('data_image.csv', index=False, header=True)

if __name__ == "__main__":
    # Set up argparse for command-line argument handling
    parser = argparse.ArgumentParser(description="Extract images from JSON and copy them to a specified directory.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing image URLs and SKU information.")
    parser.add_argument("source_dir", type=str, help="Path to the source directory where the images are located.")
    parser.add_argument("destination_dir", type=str, help="Path to the destination directory where images will be copied.")

    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.json_file, args.source_dir, args.destination_dir)
