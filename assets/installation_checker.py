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
                f"Error: Current working directory is not on the default system drive ({system_drive}). Please move Applio in the correct drive."
            )
    except:
        pass
    else:
        if "OneDrive" in now_dir:
            raise InstallationError(
                "Error: Current working directory is on OneDrive. Please move Applio in another folder."
            )
        elif " " in now_dir:
            raise InstallationError(
                "Error: Current working directory contains spaces. Please move Applio in another folder."
            )
        try:
            now_dir.encode("ascii")
        except UnicodeEncodeError:
            raise InstallationError(
                "Error: Current working directory contains non-ASCII characters. Please move Applio in another folder."
            )
