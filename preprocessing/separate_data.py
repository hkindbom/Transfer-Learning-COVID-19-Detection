"""
DD2424 Deep Learning in Data Science
May 2020
This script was written to separate data used in the COVID-Net experiment from the 
updated dataset in the corresponding git hub repo. It also contains code to move the images 
to a corresponding class partitioned directory. 
"""

# Modules
import matplotlib.pyplot as plt
import glob
import os
import cv2
import numpy as np

# Directory paths
paper_test_data_description = 'data/paper_dataset_specifications/test_COVIDx2.txt'
paper_train_data_description = 'data/paper_dataset_specifications/train_COVIDx2.txt'
current_test_data_description = 'data/test_split_v3.txt'
current_train_data_description = 'data/train_split_v3.txt'

# Get lines
def get_file_lines(filename):
    with open(filename) as f:
        content = f.readlines()
    file_lines = [x.strip() for x in content]
    return file_lines

# Identify differences between the data description used in original experiment and updated data description
def filter_out_difference(correct_data_description, wrong_data_description):
    correct_file_lines = get_file_lines(correct_data_description)
    wrong_file_lines = get_file_lines(wrong_data_description)
    nr_missing_images = 0
    
    # Need to check substrings as wrong_file_description has longer lines
    for correct_line in correct_file_lines:
        if not any(correct_line in line for line in wrong_file_lines):
            nr_missing_images += 1
                
    if nr_missing_images > 0:
        print('you are missing: ', nr_missing_images, 'images')
        return
    print('you have all images in ', correct_data_description)
    
    new_correct_upd_lines = []
    
    for wrong_line in wrong_file_lines:
        for correct_line in correct_file_lines:
            if correct_line in wrong_line:
                new_correct_upd_lines.append(wrong_line)
                   
    return new_correct_upd_lines

# Write to file
def write_line_list_to_file(file, line_list):
    with open(file, 'w') as filehandle:
        filehandle.writelines("%s\n" % line for line in line_list)

# Moving images to class partitioned directory
def get_label(img_desc_list):
    mapping = {
        'normal': 0,
        'pneumonia': 1,
        'COVID-19': 2
        }
    for class_name in mapping:
        if class_name in img_desc_list:
            return mapping[class_name]

def moving_images_to_correct_dir(): 
    
    upd_test_set_list = filter_out_difference(paper_test_data_description, current_test_data_description)
    upd_train_set_list = filter_out_difference(paper_train_data_description, current_train_data_description)

    current_dir = 'data/dataset/test/test_data'
    new_dir = 'data/dataset/test/class'
    img_descriptions = get_file_lines('data/correct_test_split.txt')
    data_path = os.path.join(current_dir, '*g')
    images = glob.glob(data_path)

    for image_path in images:
        image_name = image_path.replace(current_dir + '/', '')
        for img_desc in img_descriptions:
            if image_name in img_desc:
                img_desc_list = img_desc.split()
                label = get_label(img_desc_list)
                os.rename(image_path, new_dir +str(label) + '/' + image_name)