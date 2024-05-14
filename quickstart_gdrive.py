from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pydrive2.files import GoogleDriveFile

from src.constants import SECRETS_PATH


GoogleAuth.DEFAULT_SETTINGS["client_config_file"] = SECRETS_PATH / "pydrive_credentials.json"
gauth = GoogleAuth(settings_file=SECRETS_PATH / "pydrive_settings.yaml")
drive = GoogleDrive(gauth)

# print first 10 files
file_list = drive.ListFile({"q": "'root' in parents and trashed=false"}).GetList()
for file1 in file_list:
    print("title: %s, id: %s" % (file1["title"], file1["id"]))
