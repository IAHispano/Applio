import json
import os
import gc
import psutil
import requests
import subprocess
import time
import logging
import sys
import shutil
now_dir = os.getcwd()
sys.path.append(now_dir)
first_cell_executed = False
file_folder = "Colab-for-MDX_B"
def first_cell_ran():
    global first_cell_executed
    if first_cell_executed:
        #print("The 'first_cell_ran' function has already been executed.")
        return

    

    first_cell_executed = True
    os.makedirs("tmp_models", exist_ok=True)



    class hide_opt:  # hide outputs
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    def get_size(bytes, suffix="B"):  # read ram
        global svmem
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if bytes < factor:
                return f"{bytes:.2f}{unit}{suffix}"
            bytes /= factor
        svmem = psutil.virtual_memory()


    def use_uvr_without_saving():
        print("Notice: files won't be saved to personal drive.")
        print(f"Downloading {file_folder}...", end=" ")
        with hide_opt():
            #os.chdir(mounting_path)
            items_to_move = ["demucs", "diffq","julius","model","separated","tracks","mdx.py","MDX-Net_Colab.ipynb"]
            subprocess.run(["git", "clone", "https://github.com/NaJeongMo/Colab-for-MDX_B.git"])
            for item_name in items_to_move:
                item_path = os.path.join(file_folder, item_name)
                if os.path.exists(item_path):
                    if os.path.isfile(item_path):
                        shutil.move(item_path, now_dir)
                    elif os.path.isdir(item_path):
                        shutil.move(item_path, now_dir)
            try:
                shutil.rmtree(file_folder)
            except PermissionError:
                print(f"No se pudo eliminar la carpeta {file_folder}. Puede estar relacionada con Git.")

    
    use_uvr_without_saving()
    print("done!")
    if not os.path.exists("tracks"):
        os.mkdir("tracks")
first_cell_ran()