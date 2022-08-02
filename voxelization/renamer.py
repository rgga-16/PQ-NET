import os 

root = './partnet_raw/Chair' 
for dir in os.listdir(root):
    dirpath = os.path.join(root,dir)

    files = os.listdir(dirpath)
    for i in range(len(files)):
        file = files[i]
        oldpath = os.path.join(dirpath,file)
        newpath = os.path.join(dirpath,f'object_{i+1}.obj')
        os.rename(oldpath,newpath)
    