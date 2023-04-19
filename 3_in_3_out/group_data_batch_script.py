import os
import re

def shift_file_name(directory, shift_amount):
    # get a list of all files in the directory
    file_list = os.listdir(directory)

    # loop through the list of files
    for filename in file_list:
        # search for the pattern in the filename
        match = re.search(r'\d+', filename)
        if match:
            # get the start and end indices of the matched number
            start, end = match.span()

            # extract the number and convert it to an int
            number = int(filename[start:end])

            # add the shift amount to the number
            new_number = str(number + shift_amount)

            # create the new filename by replacing the old number with the new one
            new_filename = filename[:start] + new_number + filename[end:]

            # construct the full path to the old and new files
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)

            # rename the file
            os.rename(old_file_path, new_file_path)


dir = "D:/archivos_locales/TFG/datasets/tracknetV2_padel/npy_4folds_occluded/validation/"

file_list = os.listdir(dir)
current_shift = 0

# loop through the list of files
for folder in file_list:
    if os.path.isdir(dir + folder):
        
        current_dir = dir + folder + "/data"

        frames = os.listdir(current_dir)
       
        for i in range(len(frames)):
            frames[i] = int(frames[i][7:-4])

        shift_file_name(current_dir, current_shift)

        current_shift += max(frames)