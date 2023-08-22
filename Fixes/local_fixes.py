import os
import sys
import time
import shutil
import requests
import zipfile

file_name2 = 'go-web.bat'
text_to_insert2 = """python infer-web.py --pycmd runtime\python.exe --port 7897
pause"""

with open(file_name2, 'w') as archivo:
        archivo.write(text_to_insert2)
print(f"Se ha modificado el contenido de '{file_name2}'.")

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
                    print('¡Ya estaba arreglado! Se salta añadir una línea...')
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

if __name__ == "__main__":
    current_path = os.getcwd()
    file_name = 'extract_f0_print.py'
    line_to_find = 'import numpy as np, logging'
    text_to_insert = "sys.path.append(r'" + current_path + "')"
    

    success_1 = insert_new_line(file_name, line_to_find, text_to_insert)
    if success_1:
        print('¡La primera operación fue un éxito!')
    else:
        print('¡Se saltó la primera operación porque ya estaba arreglada!')

    file_name = 'infer-web.py'
    old_text = 'with gr.Blocks(theme=gr.themes.Soft()) as app:'
    new_text = 'with gr.Blocks() as app:'

    success_2 = replace_in_file(file_name, old_text, new_text)
    if success_2:
        print('¡La segunda operación fue un éxito!')
    else:
        print('¡La segunda operación se omitió porque ya estaba arreglada!')

    print('¡Correcciones locales exitosas! Ahora debería poder inferir y entrenar localmente en Applio RVC Fork.')
    
    time.sleep(5)

def find_torchcrepe_directory(directory):
    """
    Busca recursivamente la carpeta de mayor jerarquía denominada 'torchcrepe' dentro de un directorio.
    Devuelve la ruta del directorio encontrado o Ninguno si no se encuentra.
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
            print("Saltando la descarga de torchcrepe. La carpeta ya existe.")
            return

        # Download the file
        print("Iniciando la descarga de torchcrepe...")
        response = requests.get(url)

        # Raise an error if the GET request was unsuccessful
        response.raise_for_status()
        print("Descarga finalizada.")

        # Save the downloaded file
        zip_file_path = os.path.join(temp_dir, 'master.zip')
        os.makedirs(temp_dir, exist_ok=True)
        with open(zip_file_path, 'wb') as file:
            file.write(response.content)
        print(f"Archivo zip guardado en {zip_file_path}")

        # Extract the zip file
        print("Extrayendo contenidos...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            zip_file.extractall(temp_dir)
        print("Extracción finalizada.")

        # Locate the torchcrepe folder and move it to the destination directory
        torchcrepe_dir = find_torchcrepe_directory(temp_dir)
        if torchcrepe_dir:
            shutil.move(torchcrepe_dir, destination_dir)
            print(f"Se movió el directorio torchcrepe a {destination_dir}!")
        else:
            print("No se pudo localizar el directorio de torchcrepe.")

    except Exception as e:
        print("Torchcrepe no descargado con éxito?", e)

    # Clean up temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

# Run the function
download_and_extract_torchcrepe()

temp_dir = 'temp_torchcrepe'

if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
