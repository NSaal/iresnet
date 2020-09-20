import os
path = 'I:/datasets/image-net/train'
filedict = {}
for lists in os.listdir(path):
    sub_path = path+'/'+lists
    filenum = 0
    for files in os.listdir(sub_path):
        files_name = sub_path+'/'+files
        if os.path.isfile(files_name):
            filenum += 1
    if filenum != 1300:
        filedict[lists] = filenum
print(len(filedict))
path2 = 'D:/dataset/imagenet/train'
for lists in os.listdir(path2):
    sub_path = path2+'/'+lists
    filenum = 0
    for files in os.listdir(sub_path):
        files_name = sub_path+'/'+files
        if os.path.isfile(files_name):
            filenum += 1
    if filenum != 1300 and filedict.get(lists, "None") != filenum:
        print(files)
        print(filedict[files])
        print(filenum)
