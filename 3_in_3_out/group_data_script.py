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

# test the function
shift_file_name("D:/archivos_locales/TFG/datasets/training/data4", 5222)