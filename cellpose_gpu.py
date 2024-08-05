import numpy as np
from cellpose import models
from PIL import Image
import glob as glob
import os
from tqdm import tqdm
import shutil
import random
import time
import cv2
from typing import Set
import pickle
import matplotlib.pyplot as plt


# util funcs
def copy_random_png_images(source_dir: str, target_dir: str, percentage: int) -> None:
    """
    Randomly copy percentage% png images from source directory/folder, including the sub folders to target directory
    :param source_dir:  source directory/folder
    :param target_dir:  target directory/folder
    :param percentage:  an integer from [0, 100]
    :return:
    """

    for root, dirs, files in os.walk(source_dir):
        for subfolder in dirs:
            source_subfolder = os.path.join(root, subfolder)
            target_subfolder = os.path.join(target_dir, os.path.relpath(source_subfolder, source_dir))
            os.makedirs(target_subfolder, exist_ok=True)

            # List all .png files in the source subfolder
            png_files = [file for file in os.listdir(source_subfolder) if file.lower().endswith(".png")]

            # Calculate the number of .png files to keep
            num_png_files_to_keep = int(len(png_files) * percentage / 100)

            # Randomly select .png files to copy
            png_files_to_copy = random.sample(png_files, num_png_files_to_keep)

            # Copy selected .png files to the target subfolder
            for png_file_to_copy in png_files_to_copy:
                source_png_file = os.path.join(source_subfolder, png_file_to_copy)
                target_png_file = os.path.join(target_subfolder, png_file_to_copy)
                shutil.copy2(source_png_file, target_png_file)


def find_files(directory: str, format: str = '.svs') -> Set[str]:
    """
    Find all folders/subdirectories that contain files with the specified format.
    :param directory: The root directory to search for files.
    :param format: The type of the files to search for (default is '.svs', for the WSI file search).
    :return: A set of directories containing files with the specified format
    """
    svs_directories = set()  # Use a set to store unique parent directories

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(format):
                svs_directories.add(root)  # Use add() to add unique directories

    return svs_directories


class CellposeProcessor:

    def __init__(self, use_gpu=True, model_type="nuclei"):
        self.gpu = use_gpu
        self.model = models.Cellpose(model_type=model_type, gpu=use_gpu)

    def model_eval(self, image_array,
                   channels=[0, 0],
                   diameter=None,
                   flow_threshold=0.4,
                   min_size=15,
                   invert=True):

        """
        reference: cellpose.models.Cellpose.eval() method

        :param  image_array:
        :param  channels: 0=grayscale, 1=red, 2=green, 3=blue. (0=None, will set to zero)
        :param  diameter: (30 pixels in the cyto model and 17 pixels in the case of the nuclei model)
        :param  flow_threshold: the maximum allowed error of the flows for each mask. The default is 0.4
        :param  min_size: minimum number of pixels per mask, can turn off with -1 (default 15)
        other params （default）/customized by Cellpose.eval()

        :return:
        Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                    labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = XY flows at each pixel
                flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
                flows[k][3] = final pixel locations after Euler integration

            styles: list of 1D arrays of length 256, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

            diams: list of diameters, or float (if do_3D=True)
        """

        return self.model.eval(image_array, channels=channels, diameter=diameter, flow_threshold=flow_threshold,
                               min_size=min_size, invert=invert)

    def load_image(self, img_path):

        image = np.array(Image.open(img_path).convert('RGB'))
        return image


    def save_plots_grayscale(self, image_array, output_dir):
        """
        :param image_array: image as array
        :param output_dir: complete path for output, e.g., /xxxx_instance.png
        :param cmap_matplotlib:
        :return:
        """

        # binary map
        if image_array.ndim != 3:
            image = Image.fromarray(np.stack((image_array, image_array, image_array), axis=-1))
            image.save(output_dir)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def inference_instance_loop(self, image_files, output_dir, suffix):

        """
        Inference all *.png files  (image_files, a list of PNG file path). Save results (e.g., *_instance.png) to
        output_dir
        :param image_files: list of directories ['/directory/to/1.png', '/directory/to/2.png', ... ]
        :return:
        """
        for image_file in image_files:
            try:
                image = self.load_image(image_file)
                mask, flows, _, _ = self.model_eval(image, channels=[0, 0], diameter=None, flow_threshold=0.8, min_size=15,
                                                    invert=True)

                basename = os.path.basename(image_file)

                if isinstance(suffix, str) and 'binary' in suffix:
                    output = os.path.join(output_dir, basename.replace(".png", suffix))
                    self.save_plots_grayscale(mask, output)

                elif isinstance(suffix[0], str):
                    for s in suffix:
                        output = os.path.join(output_dir, basename.replace(".png", s))

                        # cell probability map
                        if s == '_cellprob.npy':
                            mask_nonzero = np.where(mask > 0, 1, 0)
                            P = self._sigmoid(flows[2]) * mask_nonzero

                            np.save(output.replace(".png", ".npy"), P)

                        # save binary map
                        if s == '_binary.png':
                            binary_map = (mask > 0).astype(np.uint8)
                            self.save_plots_grayscale(binary_map * 255, output)

                        # save contours (png) and instance map (npy)
                        if s == '_contours.png':

                            unique_labels = np.unique(mask)
                            unique_labels = unique_labels[unique_labels != 0]

                            # get contours of instance predictions
                            for label in unique_labels:
                                binary_mask = np.where(mask == label, 255, 0).astype(np.uint8)
                                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
                            Image.fromarray(image).save(output)

                            # save instance map as .npy File
                            np.save(output.replace(".png", ".npy"), mask)

            except Exception as e:
                print(f'{image_file} has IO error')


if __name__ == "__main__":
    # path_to_patch_folder is the root directory of all patch (png) folder
    # absolute_input_path/relative_path: path/to/png_folders
    absolute_input_path = '/home/guoj5/Desktop/wsi-select/Version2_patch_sampled/rodent_kidney_images'
    relative_input_path = '4-ADR associated mouse FSGS-2022'
    absolute_output_path = '/home/guoj5/Desktop/wsi-select/Version2_patch_sampled_predictions/cellpose_pred/rodent_kidney_images'
    path_to_patch_folder = os.path.join(absolute_input_path, relative_input_path)

    # find all folders of .png (patch) in input path
    if not os.path.exists(relative_input_path + '_data_dirs.pkl'):
        data_dirs = list(find_files(path_to_patch_folder, format='.png'))
        with open(relative_input_path + '_data_dirs.pkl', 'wb') as binary_file:
            pickle.dump(data_dirs, binary_file)
    else:
        with open(relative_input_path + '_data_dirs.pkl', 'rb') as file:
            data_dirs = pickle.load(file)


    # model
    cp = CellposeProcessor(use_gpu=True, model_type="nuclei")
    cp.time_start = time.time()

    resume_dataset = 0
    # inference each folder, and save to absolute_output_path with original structure
    for i in tqdm(range(resume_dataset, len(data_dirs))):
        data_dir = data_dirs[i]
        output_dir = os.path.join(absolute_output_path, data_dir[data_dir.find(relative_input_path):])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else: continue

        png_files = glob.glob(os.path.join(data_dir, '*.png'))
        image_files = [file for file in png_files if 'mask' not in file]

        cp.inference_instance_loop(image_files=image_files, output_dir=output_dir,
                                   suffix=['_contours.png'])

    cp.run_time = time.time() - cp.time_start
    print(f'run time is {cp.run_time}')
