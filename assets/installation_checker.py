import sys
import os

now_dir = os.getcwd()
sys.path.append(now_dir)


class InstallationError(Exception):
    def __init__(self, message="InstallationError"):
        self.message = message
        super().__init__(self.message)


def check_installation():
    try:
        system_drive = os.getenv("SystemDrive")
        current_drive = os.path.splitdrive(now_dir)[0]
        if current_drive.upper() != system_drive.upper():
            raise InstallationError(
                f"Installation Error: The current working directory is on drive {current_drive}, but the default system drive is {system_drive}. Please move Applio to the {system_drive} drive."
            )
    except:
        pass
    else:
        if "OneDrive" in now_dir:
            raise InstallationError(
                "Installation Error: The current working directory is located in OneDrive. Please move Applio to a different folder."
            )
        elif " " in now_dir:
            raise InstallationError(
                "Installation Error: The current working directory contains spaces. Please move Applio to a folder without spaces in its path."
            )
        try:
            now_dir.encode("ascii")
        except UnicodeEncodeError:
            raise InstallationError(
                "Installation Error: The current working directory contains non-ASCII characters. Please move Applio to a folder with only ASCII characters in its path."
            )
