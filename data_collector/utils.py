import serial 
import serial.tools.list_ports 
import sys
import re
import time
import hashlib
import os
import shutil

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def welcome():
    print("""
   _____       _ _        _           _ _   _______        _       _             
  / ____|     (_) |      | |         | | | |__   __|      (_)     (_)            
 | (___  _ __  _| | _____| |__   __ _| | |    | |_ __ __ _ _ _ __  _ _ __   __ _ 
  \___ \| '_ \| | |/ / _ \ '_ \ / _` | | |    | | '__/ _` | | '_ \| | '_ \ / _` |
  ____) | |_) | |   <  __/ |_) | (_| | | |    | | | | (_| | | | | | | | | | (_| |
 |_____/| .__/|_|_|\_\___|_.__/ \__,_|_|_|    |_|_|  \__,_|_|_| |_|_|_| |_|\__, |
        | |                                                                 __/ |
        |_|                                                                |___/ 
 """)

def current_milli_time():
    return round(time.time() * 1000)

def remove_non_numbers(input_string):
    return re.sub(r'\D', '', input_string)


def delete_last_line():
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")

def delete_multiple_lines(number):
    for i in range(number):
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")

def print_hit(index, duration, samples, end_reason, label, reprint=False):
    if label == "rim":
        print(bcolors.WARNING, end="")
    elif label == "net":
        print(bcolors.HEADER, end="")
    elif label == "discard":
        print(bcolors.FAIL, end="")
    elif label == "rollup":
        print(bcolors.OKCYAN, end="")
    if reprint:
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
    if label != "discard":
        print("#" + str(index).center(15) + "|" + (str(duration) + "ms").center(15) +
                                "|" + str(samples).center(15) + "|" + end_reason.center(15) + 
                                "|" + label.upper().center(15) + "#")
    else:
        print(strike(str(index).center(15) + "|" + (str(duration) + "ms").center(15) +
                                "|" + str(samples).center(15) + "|" + end_reason.center(15) + 
                                "|" + label.upper().center(15) + "#"))
    print(bcolors.HEADER, end="")
    
def strike(text):
    return ''.join([u'\u0336{}'.format(c) for c in text])

def connect_to_arduino():
    comports = list(serial.tools.list_ports.comports())

    selected_port = None
    for i, (port, desc, hwid) in enumerate(comports):
        if "Nano 33 BLE" in desc:
            selected_port = i
            print(bcolors.OKGREEN + "Arduino Nano 33 BLE detected & connected on port {}: {}".format(i, port))
            break
            


    # If no Arduino Nano 33 BLE was found, ask the user to select a port
    if selected_port is None:
        for i, (port, desc, hwid) in enumerate(comports):
            print("{}: {} [{}]".format(i, port, desc, hwid))
        print("Please select a port.")
        selected_port = int(input("Select a port number: "))

    # Get the port path
    port_path = comports[selected_port][0]

    # Open the selected port
    try:
        ser = serial.Serial(port_path, 250000, timeout=1)
        print("Serially connected with " + ser.name)
        print(bcolors.HEADER)
        return ser
    except Exception as e:
        print("Failed to open port {}".format(port_path))
        print("Error: {}".format(e))
        return None


def collect_and_flatten(root_dir, output_dir):
    """
    Iterates through folders, calculates MD5 hashes, and collects unique
    files into a categorized output directory with ascending filenames.
    Tracks input, output, and duplicate file counts.

    Args:
        root_dir (str): The root directory to search within.
        output_dir (str): The directory where collected data will be stored.
    """

    seen_hashes = {}  # Store MD5 hashes and corresponding filenames
    file_counts = {}  # Store file counts for each category (subdirectory) in output
    input_counts = {} # Store file counts for each category in the input
    duplicate_count = 0 # Total number of duplicate files found

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for date_dir in os.listdir(root_dir):
        date_path = os.path.join(root_dir, date_dir)

        # Only process directories (ignore files directly under root)
        if not os.path.isdir(date_path):
            continue

        for subdir in ["discard", "net", "rim", "rollup"]:  # Iterate through subdirectories
            subdir_path = os.path.join(date_path, subdir)

            # Check if the subdirectory exists
            if not os.path.exists(subdir_path):
                continue

            #Initialize counts if they don't exist.
            if subdir not in input_counts:
                input_counts[subdir] = 0

            # Ensure category subfolder exists in output
            category_path = os.path.join(output_dir, subdir)
            if not os.path.exists(category_path):
                os.makedirs(category_path)
                file_counts[subdir] = 0  # Initialize count if new

            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)

                # Ensure only processing files.
                if not os.path.isfile(file_path):
                    continue

                input_counts[subdir] += 1 #Increment each time.

                # Calculate MD5 hash
                with open(file_path, "rb") as f:
                    file_contents = f.read()
                    md5_hash = hashlib.md5(file_contents).hexdigest()

                if md5_hash in seen_hashes:
                    print(f"WARNING: Duplicate file found! {filename} (from {subdir_path}) has the same hash as {seen_hashes[md5_hash]}. Ignoring.")
                    duplicate_count += 1
                else:
                    # Collect the file
                    file_counts[subdir] += 1
                    new_filename = f"{file_counts[subdir] - 1}.csv"  # Start at 0
                    new_file_path = os.path.join(category_path, new_filename)
                    shutil.copy2(file_path, new_file_path)  #copy2 preserves metadata
                    
                    # Update hash and filename in dictionary
                    seen_hashes[md5_hash] = new_file_path
                    print(f"Collected {filename} to {new_file_path}")

    print("\n--- Summary ---")
    print("Input Data:")
    for category, count in input_counts.items():
        print(f"  {category}: {count} files")

    print("\nOutput Data:")
    for category, count in file_counts.items():
        print(f"  {category}: {count} files")

    print(f"\nDuplicate Files Found: {duplicate_count}")

def count_data_categories(root_dir):
    """
    Counts the number of files in specific subdirectories within a root directory.

    Args:
        root_dir (str): The root directory to search within.

    Returns:
        dict: A dictionary where keys are category names ("discard", "net", "rim", "rollup")
              and values are the corresponding file counts.
    """

    input_counts = {}  # Store file counts for each category in the input

    for date_dir in os.listdir(root_dir):
        date_path = os.path.join(root_dir, date_dir)

        # Only process directories (ignore files directly under root)
        if not os.path.isdir(date_path):
            continue

        for subdir in ["discard", "net", "rim", "rollup"]:  # Iterate through subdirectories
            subdir_path = os.path.join(date_path, subdir)

            # Check if the subdirectory exists
            if not os.path.exists(subdir_path):
                continue

            # Initialize counts if they don't exist.
            if subdir not in input_counts:
                input_counts[subdir] = 0

            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)

                # Ensure only processing files.
                if not os.path.isfile(file_path):
                    continue

                input_counts[subdir] += 1  # Increment each time.

    print("\n--- Summary ---")
    print("Recorded Data counts:")
    for category, count in input_counts.items():
        print(f"  {category}: {count} files")
    return input_counts