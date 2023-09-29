import os
import time
import shutil
import requests
import zipfile

def insert_new_line(file_name, line_to_find, text_to_insert):
    lines = []
    with open(file_name, 'r', encoding='utf-8') as read_obj:
        lines = read_obj.readlines()
    already_exists = False
    with open(file_name + '.tmp', 'w', encoding='utf-8') as write_obj:
        for i in range(len(lines)):
            write_obj.write(lines[i])
            if lines[i].strip() == line_to_find:
                # If next line exists and starts with sys.path.append, skip
                if i+1 < len(lines) and lines[i+1].strip().startswith("sys.path.append"):
                    print('It was already fixed! Skip adding a line...')
                    already_exists = True
                    break
                else:
                    write_obj.write(text_to_insert + '\n')
    # If no existing sys.path.append line was found, replace the original file
    if not already_exists:
        os.replace(file_name + '.tmp', file_name)
        return True
    else:
        # If existing line was found, delete temporary file
        os.remove(file_name + '.tmp')
        return False

def replace_in_file(file_name, old_text, new_text):
    with open(file_name, 'r', encoding='utf-8') as file:
        file_contents = file.read()

    if old_text in file_contents:
        file_contents = file_contents.replace(old_text, new_text)
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(file_contents)
            return True

    return False


def find_torchcrepe_directory(directory):
    """
    Recursively searches for the topmost folder named 'torchcrepe' within a directory.
    Returns the path of the directory found or None if none is found.
    """
    for root, dirs, files in os.walk(directory):
        if 'torchcrepe' in dirs:
            return os.path.join(root, 'torchcrepe')
    return None

def download_and_extract_torchcrepe():
    url = 'https://github.com/maxrmorrison/torchcrepe/archive/refs/heads/master.zip'
    temp_dir = 'temp_torchcrepe'
    destination_dir = os.getcwd()

    try:
        torchcrepe_dir_path = os.path.join(destination_dir, 'torchcrepe')

        if os.path.exists(torchcrepe_dir_path):
            print("Skipping the torchcrepe download. The folder already exists.")
            return

        # Download the file
        print("Starting torchcrepe download...")
        response = requests.get(url)

        # Raise an error if the GET request was unsuccessful
        response.raise_for_status()
        print("Download completed.")

        # Save the downloaded file
        zip_file_path = os.path.join(temp_dir, 'master.zip')
        os.makedirs(temp_dir, exist_ok=True)
        with open(zip_file_path, 'wb') as file:
            file.write(response.content)
        print(f"Zip file saved to {zip_file_path}")

        # Extract the zip file
        print("Extracting content...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            zip_file.extractall(temp_dir)
        print("Extraction completed.")

        # Locate the torchcrepe folder and move it to the destination directory
        torchcrepe_dir = find_torchcrepe_directory(temp_dir)
        if torchcrepe_dir:
            shutil.move(torchcrepe_dir, destination_dir)
            print(f"Moved the torchcrepe directory to {destination_dir}!")
        else:
            print("The torchcrepe directory could not be located.")

    except Exception as e:
        print("Torchcrepe not successfully downloaded", e)

    # Clean up temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

# Run the function
download_and_extract_torchcrepe()

temp_dir = 'temp_torchcrepe'

if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
