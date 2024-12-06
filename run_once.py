import os
import subprocess
import shutil
from PIL import Image, ImageChops


'''
This file was made because I had a lot of trouble running the code on the cloud GPUs @ LamdaLabs,
as I wanted to use the newest PyTorch version, but the cloud GPUs only supported PyTorch 2.4.1.
I messed up a lot of instances, and for each time I had to setup a new instance.
'''
def crop_black_borders(image_path: str, counter: int):
    image = Image.open(image_path)

    # Get the bounding box of the non-black region
    bounding_box = Image.new(image.mode, image.size, (0, 0, 0))
    diff = ImageChops.difference(image, bounding_box)
    bbox = diff.getbbox()

    if bbox:
        cropped_img = image.crop(bbox)
        return cropped_img, (counter + 1)
    else:
        return image, counter


def crop_borders_all_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    counter = 0
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        output_folder_path = os.path.join(output_dir, folder)

        if os.path.isdir(folder_path):
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    file_path = os.path.join(folder_path, file)
                    output_file_path = os.path.join(output_folder_path, file)

                    cropped_image, counter = crop_black_borders(file_path, counter)
                    cropped_image.save(output_file_path)

                    if counter % 10 == 0:
                        print(f"{counter} images cropped")

    print(f"{counter} images cropped")


def create_folders_download_dataset():
    # Create directories
    base_dir = os.path.expanduser("~/Documents")
    dirs_to_create = [
        base_dir,
        os.path.join(base_dir, "IKT450"),
        os.path.join(base_dir, "Datasets"),
        os.path.join(base_dir, "Datasets/Fish_GT")
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Get Fish Ground Truth dataset
    dataset_dir = os.path.join(base_dir, "Datasets/Fish_GT")
    os.chdir(dataset_dir)
    print(f"Changed directory to: {dataset_dir}")

    # Download files
    dataset_files = [
        "https://homepages.inf.ed.ac.uk/rbf/fish4knowledge/GROUNDTRUTH/RECOG/class_id.csv",
        "https://homepages.inf.ed.ac.uk/rbf/fish4knowledge/GROUNDTRUTH/RECOG/Archive/fishRecognition_GT.tar"
    ]

    for file_url in dataset_files:
        subprocess.run(["wget", file_url], check=True)
        print(f"Downloaded: {file_url}")

    # Extract the tar file
    tar_file = os.path.join(dataset_dir, "fishRecognition_GT.tar")
    subprocess.run(["tar", "-xvf", tar_file], check=True)
    print(f"Extracted: {tar_file}")


def manipulate_images():
    input_dir = os.path.expanduser("~/Documents/Datasets/Fish_GT/fish_image")
    output_dir = os.path.expanduser("~/Documents/Datasets/Fish_GT/image_cropped")

    crop_borders_all_images(input_dir, output_dir)


def copy_files():
    source_file = os.path.expanduser("~/Documents/Datasets/Fish_GT/class_id.csv")
    destination_dir = os.path.expanduser("~/Documents/Datasets/Fish_GT/image_cropped")

    os.makedirs(destination_dir, exist_ok=True)
    shutil.copy(source_file, os.path.join(destination_dir, "class_id.csv"))

    print(f"File copied to {destination_dir}")


def run_once():
    create_folders_download_dataset()
    manipulate_images()
    copy_files()


run_once()

