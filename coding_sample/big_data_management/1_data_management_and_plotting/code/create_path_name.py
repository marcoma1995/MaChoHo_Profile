import os

def create_path_name(folder_name, file_name):
    """Create relative path with folder name and file name.
    
    folder_name -- string including folder name of the destination
    file_name -- string including file name of the destination
    
    """
    
    cwd = os.getcwd()
    cwd_2 = cwd[:-4]
    cwd_3 = cwd_2 +"/{}".format(folder_name)+"/{}".format(file_name)
    cwd_4 = cwd_3.replace("\\","/")
    
    return cwd_4