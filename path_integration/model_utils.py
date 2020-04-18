import os
import glob

def get_latest_model_file(folder):
    """
    returns the location of the newest file in the provided foler path
    note: folder path NEEDS to end on a /

    returns None if no files were found in given location
    """
    files = glob.glob(folder + 'model_epoch_*.pt')
    if files == []:
        print("No files found.")
        return None
    newest_file = max(files, key=get_model_epoch)
    return newest_file

def get_model_epoch(file):
    """
    returns the epoch of the provided model file
    """
    n = file.split('_')[-1].split('.')[0]
    return int(n)
