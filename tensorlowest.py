from tensorboard.backend.event_processing import event_accumulator

import os
from shutil import copy2
from re import search as RSearch
import pandas as pd
from ast import literal_eval as LEval

weights_dir = 'weights/'

def find_biggest_tensorboard(tensordir):
    try:
        files = [f for f in os.listdir(tensordir) if f.endswith('.0')]
        if not files:
            print("No files with the '.0' extension found!")
            return

        max_size = 0
        biggest_file = ""

        for file in files:
            file_path = os.path.join(tensordir, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > max_size:
                    max_size = file_size
                    biggest_file = file

        return biggest_file

    except FileNotFoundError:
        print("Couldn't find your model!")
        return

def main(model_name, save_freq, lastmdls):
    global lowestval_weight_dir, scl

    tensordir = os.path.join('logs', model_name)
    lowestval_weight_dir = os.path.join(tensordir, "lowestvals")
    
    latest_file = find_biggest_tensorboard(tensordir)
    
    if latest_file is None:
        print("Couldn't find a valid tensorboard file!")
        return
    
    tfile = os.path.join(tensordir, latest_file)
    
    ea = event_accumulator.EventAccumulator(tfile,
        size_guidance={
        event_accumulator.COMPRESSED_HISTOGRAMS: 500,
        event_accumulator.IMAGES: 4,
        event_accumulator.AUDIO: 4,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1,
    })

    ea.Reload()
    ea.Tags()

    scl = ea.Scalars('loss/g/total')

    listwstep = {}
    
    for val in scl:
        if (val.step // save_freq) * save_freq in [val.step for val in scl]:
            listwstep[float(val.value)] = (val.step // save_freq) * save_freq

    lowest_vals = sorted(listwstep.keys())[:lastmdls]

    sorted_dict = {value: step for value, step in listwstep.items() if value in lowest_vals}
    
    return sorted_dict

def selectweights(model_name, file_dict, weights_dir, lowestval_weight_dir):
    os.makedirs(lowestval_weight_dir, exist_ok=True)
    logdir = []
    files = []
    lbldict = {
        'Values': {},
        'Names': {}
    }
    weights_dir_path = os.path.join(weights_dir, "")
    low_val_path = os.path.join(os.getcwd(), os.path.join(lowestval_weight_dir, ""))
    
    try:
        file_dict = LEval(file_dict)
    except Exception as e: 
        print(f"Error! {e}")
        return f"Couldn't load tensorboard file! {e}"
    
    weights = [f for f in os.scandir(weights_dir)]
    for key, value in file_dict.items():
        pattern = fr"^{model_name}_.*_s{value}\.pth$"
        matching_weights = [f.name for f in weights if f.is_file() and RSearch(pattern, f.name)]
        for weight in matching_weights:
            source_path = weights_dir_path + weight
            destination_path = os.path.join(lowestval_weight_dir, weight)
            
            copy2(source_path, destination_path)

            logdir.append(f"File = {weight} Value: {key}, Step: {value}")

            lbldict['Names'][weight] = weight
            lbldict['Values'][weight] = key

            files.append(low_val_path + weight)

            print(f"File = {weight} Value: {key}, Step: {value}")

            yield ('\n'.join(logdir), files, pd.DataFrame(lbldict))
            

    return ''.join(logdir), files, pd.DataFrame(lbldict)
    

if __name__ == "__main__":
    model = str(input("Enter the name of the model: "))
    sav_freq = int(input("Enter save frequency of the model: "))
    ds = main(model, sav_freq)
    
    if ds: selectweights(model, ds, weights_dir, lowestval_weight_dir)
    