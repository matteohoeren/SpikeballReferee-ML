import os
import pandas as pd
import os
import shutil
import pickle
from datetime import datetime as dt
from utils import bcolors

class DataWarehouse:
    def __init__(self, root_folder, data_labels):
        self.root_folder = root_folder
        self.data_labels = data_labels
        self.measurement_index = 0
        self.saved_data = []

        if not os.path.exists(root_folder):
            os.makedirs(root_folder)
        else:
            if not os.path.isfile(root_folder + '/records.pkl'):
                print("No record file found in {}".format(root_folder))
            else:
                with open(root_folder + '/records.pkl', 'rb') as f:
                    self.saved_data = pickle.load(f)
                    self.measurement_index = len(self.saved_data)
                    print("New measurement index: " + str(self.measurement_index))
                    print("Loaded {} records from {}".format(len(self.saved_data), root_folder + '/records.pkl'))
                    data_count = 0  
                    for label in self.data_labels:
                        folder_path = os.path.join(root_folder, label)
                        for name in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, name)
                            if os.path.isfile(file_path):
                                data_count += 1
                    print("Found {} records as csv files".format(data_count))
                    if(data_count != len(self.saved_data)):
                        print(bcolors.WARNING + "WARNING: The number of records in the record file does not match the number of csv files in the data folders." + bcolors.HEADER)

        for label in self.data_labels:
            if not os.path.exists("{}/{}".format(root_folder,label)):
                os.makedirs("{}/{}".format(root_folder,label))
    
    def get_root(self):
        return self.root_folder
    
    def get_index(self):
        return self.measurement_index
    
    def get_dataset(self, index):
        for element in self.saved_data:
            if element['index'] == int(index):
                foundelement = element
        if foundelement is None or foundelement == self.saved_data[-1]:
            print("No dataset found for index {}".format(index))
            return None
        try:
            df = pd.read_csv(foundelement["filename"], header=None)
            return df.values.tolist()
        except FileNotFoundError:
            print("File {} not found.".format(foundelement["filename"]))
            return None
    
    def save_warehouse(self):
        with open(self.root_folder + '/records.pkl', 'wb') as f:
            pickle.dump(self.saved_data, f)
            print("Saved record dictionary with {} records to {}".format(len(self.saved_data), self.root_folder+ '/records.pkl'))
            

    def save_dataset(self, index, dataset, folder="net"):
        df = pd.DataFrame(dataset)
        filename = self.root_folder +"/" + folder + "/" + str(index) + ".csv"
        df.to_csv(filename, index=False, header=False)
        now = dt.now()
        self.saved_data.append({"index": index, "folder": folder, "filename": filename, "time": now.strftime("%H:%M:%S")})
        pass

    def move_last_dataset(self, target_folder):
        # Assuming self.data is the list of dictionaries
        if not self.saved_data:
            print("No data to move.")
            return

        # Get the last added dataset
        last_dataset = self.saved_data[-1]

        # Get the filename and construct the full path
        source_path = last_dataset['filename']

        # Construct the target path
        target_path = os.path.join(self.root_folder,target_folder,os.path.basename(source_path))

        # Move the file
        # print("Moving {} to {}".format(source_path, target_path))   
        shutil.move(source_path, target_path)
        self.saved_data[-1]["folder"] = target_folder
        self.saved_data[-1]["filename"] = target_path

    def move_dataset_byid(self, id, target_folder):
        # Check if the id is valid
        if id < 0 or id >= len(self.saved_data):
            print("Invalid id:" + str(id))
            return

        # Get the dataset by id
        dataset = self.saved_data[id]
        # print("Found dataset")
        # print(dataset)

        # Get the filename and construct the full path
        source_path = dataset['filename']

        # print("Source path: {}".format(source_path))
        # Construct the target path
        target_path = os.path.join(self.root_folder, target_folder, os.path.basename(source_path))
        #print("Target path: {}".format(target_path))

        # Move the file
        # print("Moving {} to {}".format(source_path, target_path))   
        shutil.move(source_path, target_path)
        self.saved_data[id]["folder"] = target_folder
        self.saved_data[id]["filename"] = target_path

        # print("modified dataset")
        # print(self.saved_data[id])

    