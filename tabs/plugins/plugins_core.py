import os, sys, shutil
import json
import gradio as gr
import zipfile
import subprocess

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

from tabs.settings.restart import restart_applio

plugins_path = os.path.join(now_dir, "tabs", "plugins", "installed")
if not os.path.exists(plugins_path):
    os.makedirs(plugins_path)
json_file_path = os.path.join(now_dir, "assets", "config.json")
current_folders = os.listdir(plugins_path)


def get_existing_folders():
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            config = json.load(file)
            return config["plugins"]
    else:
        return []


def save_existing_folders(existing_folders):
    with open(json_file_path, "r") as file:
        config = json.load(file)
        config["plugins"] = existing_folders
    with open(json_file_path, "w") as file:
        json.dump(config, file, indent=2)


def save_plugin_dropbox(dropbox):
    if "zip" not in dropbox:
        raise gr.Error(
            message="The file you dropped is not a valid plugin.zip. Please try again."
        )
    else:
        file_name = os.path.basename(dropbox)
        folder_name = file_name.split(".zip")[0]
        folder_path = os.path.join(plugins_path, folder_name)
        zip_file_path = os.path.join(plugins_path, file_name)

        if os.path.exists(folder_name):
            os.remove(folder_name)

        shutil.move(dropbox, os.path.join(plugins_path, file_name))
        print("Proceeding with the extraction...")

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(plugins_path)
        os.remove(zip_file_path)

        if os.path.exists(os.path.join(folder_path, "requirements.txt")):
            if os.name == "nt":
                subprocess.run(
                    [
                        os.path.join("env", "python.exe"),
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        os.path.join(folder_path, "requirements.txt"),
                    ]
                )
            else:
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        os.path.join(folder_path, "requirements.txt"),
                    ]
                )
        else:
            print("No requirements.txt file found in the plugin folder.")

        save_existing_folders(get_existing_folders() + [folder_name])

        print(
            f"{folder_name} plugin installed in {plugins_path}! Restarting applio to apply the changes."
        )
        gr.Info(
            f"{folder_name} plugin installed in {plugins_path}! Restarting applio to apply the changes."
        )
        restart_applio()
    return None


def check_new_folders():
    existing_folders = get_existing_folders()
    new_folders = set(current_folders) - set(existing_folders)
    save_existing_folders(current_folders)
    if new_folders:
        for new_folder in new_folders:
            complete_path = os.path.join(plugins_path, new_folder)
            print(f"New plugin {new_folder} found, installing it...")

            if os.path.exists(os.path.join(complete_path, "requirements.txt")):
                if os.name == "nt":
                    subprocess.run(
                        [
                            os.path.join("env", "python.exe"),
                            "-m",
                            "pip",
                            "install",
                            "-r",
                            os.path.join(complete_path, "requirements.txt"),
                        ]
                    )
                else:
                    subprocess.run(
                        [
                            "python",
                            "-m",
                            "pip",
                            "install",
                            "-r",
                            os.path.join(complete_path, "requirements.txt"),
                        ]
                    )
            else:
                print("No requirements.txt file found in the plugin folder.")
        print("Plugins checked and installed! Restarting applio to apply the changes.")
        restart_applio()
