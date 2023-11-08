# DVC Reference

## Set up GDrive Storage as a remote

 - Log into GDrive.
 - On the left side, create a folder structure appropriate for your
   project, using the mouse (right click to create a new folder).
 - Go to the folder which you want to use as a storage, e.g. `images`.
 - Copy the last part of the URL into the clip board. For example, if
   the URL is
   `https://drive.google.com/drive/folders/1vZ1p8Tsq_nA_bONknWJaiR2`,
   copy the last part beginning with `1vZ....`.
 - Open a shell / terminal. Go into your DVC project folder. Activate the
   python virtual environment.
 - In the project folder, add a remote by typing: `dvc remote add
   myremote gdrive://<folder-ID>` We thereby associate the URL with a
   name, so that we can have multiple storages with different URLs.
 - You can check that you have successfully added the remote by typing
   `dvc remote list`.
 - Next, tell DVC to use this remote as a default: `dvc remote default
   myremote`.
 - Now store the data you want to track in somewhere in your project.
   The folder name does not have to correspond with the folders in the
   storage.
 - Then register the data in DVC: `dvc add somefolder/someimage.jpg`.
 - Now DVC tracks the file. To push it to the repo, use `dvc push`.
   You will be asked this one time to enter your credentials in the
   webbrowser and to authorize DVC. Give it access to read and write.
   The file will then be uploaded.
 
## Modify files

```
dvc remove <filename>.dvc
<change the file>
dvd add <filename>
```

## Get an overview of all files tracked by DVC

```
dvc data status --unchanged
```

This command is supposed to be similar to `git status`. See `dvc data
status --help` for more useful options.


