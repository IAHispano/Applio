import sys
import os

sys.path.append(os.getcwd())


class InstallationError(Exception):
    def __init__(self, message="InstallationError"):
        self.message = message
        super().__init__(self.message)


def check_installation():
    try:
        system_drive = os.getenv("SystemDrive")
        current_drive = os.path.splitdrive(os.getcwd())[0]
        if current_drive.upper() != system_drive.upper():
            raise InstallationError(
                f"Error: Current working directory is not on the default system drive ({system_drive}). Please move Applio in the correct drive."
            )
    except:
        pass
    else:
        if "OneDrive" in os.getcwd():
            raise InstallationError(
                "Error: Current working directory is on OneDrive. Please move Applio in another folder."
            )
        elif " " in os.getcwd():
            raise InstallationError(
                "Error: Current working directory contains spaces. Please move Applio in another folder."
            )
        try:
            os.getcwd().encode("ascii")
        except UnicodeEncodeError:
            raise InstallationError(
                "Error: Current working directory contains non-ASCII characters. Please move Applio in another folder."
            )
