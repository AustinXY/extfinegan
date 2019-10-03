import os

mypath = '../data/ut-zap50k-images-square/'
for path, folders, files in os.walk(mypath):
    for f in files:
        os.rename(os.path.join(path, f), os.path.join(path, f.replace(' ', '_')))
    for i in range(len(folders)):
        new_name = folders[i].replace(' ', '_')
        os.rename(os.path.join(path, folders[i]), os.path.join(path, new_name))
        folders[i] = new_name