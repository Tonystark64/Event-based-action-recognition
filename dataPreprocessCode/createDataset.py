import os
import numpy as np
import pandas as pd
import cv2
import pickle

def traverse(path1, vstart, vend, output=""):
    for path, dirList, fileList in os.walk(path1):
        for fileName in fileList:
            vcount = int(path[-9:-7])
            if vcount < vstart:
                continue
            if vcount > vend:
                break
            actionNo = int(path[-2:])
            if actionNo!=4 and actionNo!=7 and actionNo!=12:
                continue
            if fileName.endswith('.avi') and path[-3]=='a':
                newName = path[-6:]+".avi"
                oldpath = os.path.join(path,fileName)
                cap = cv2.VideoCapture(oldpath)
                VideoWrite = cv2.VideoWriter(os.path.join(output, newName)
                , cv2.VideoWriter_fourcc(*'PIM1'),30, (340, 256))
                success,_ = cap.read()
                while success:
                    success,frame = cap.read()
                    try:
                        VideoWrite.write(cv2.resize(frame,(340,256),interpolation=cv2.INTER_AREA))
                    except:
                        break
                cap.release()
                VideoWrite.release()
                print(newName,":completed")

def genEventFrame(path1, path2, fps=25):
    # path1 is event file
    # path2 is output folder
    # if fps is over 60, then play speed would be slow down and still fps=60
    frame = np.zeros((260,346,1),np.uint8)
    e = pickle.load(open(path1,'rb'))
    e = np.hstack(e)
    e = e[['timestamp','x','y','polarity']]
    ts = e['timestamp'][0]
    # polarity is 1 and 0
    outputName = path1[-11:]
    outputName = outputName.replace('\\','')
    outputName = outputName.replace('.pkl','.avi')
    print(outputName)
    VideoWrite = cv2.VideoWriter(os.path.join(path2,outputName), 
                cv2.VideoWriter_fourcc(*'PIM1'),fps, (340, 256), isColor=False)
    interval = int(1e6 / fps)
    for i in range(len(e)):
        if e['timestamp'][i] < ts + interval:
            pass
        else:
            ts = e['timestamp'][i]
            # frame[frame > 127] = 255
            # frame[frame <= 127] = 0
            VideoWrite.write(frame[2:258,3:343,:])
            frame[:][:] = 0
        if e['polarity'][i]:
            frame[e['y'][i]][e['x'][i]] += 1
        else:
            frame[e['y'][i]][e['x'][i]] -= 1
    VideoWrite.write(frame)
    VideoWrite.release()

def eventTra(path1, out):
    # train dataset
    # tp = out
    # val dataset
    # vp = out.replace('train','val')
    for p,d,f in os.walk(path1):
        for fn in f:
            if fn.endswith('.pkl') and int(p[-2:])>9:
                genEventFrame(os.path.join(p,fn),out)
    file1 = out.replace('train','eventTrain.txt')
    with open(file1,'a+') as f:
        f.truncate(0)
    f1 = open(file1,'a')
    for p,d,f in os.walk(out):
        for fn in f:
            num = int(fn[fn.find('a')+1:fn.find('.')])-1
            f1.write(fn+" "+str(num)+"\n")
    f1.close()

def eventFlip(path1):
    f1 = open(os.path.join(path1,'trainVideo.txt'),'a')
    f2 = open(os.path.join(path1,'valVideo.txt'),'a')
    for p,d,f in os.walk(path1):
        for fn in f:
            if fn.endswith('.avi'):
                newfn = fn[:fn.find('.')]+'flip.avi'
                newpath = os.path.join(p,newfn)
                oldpath = os.path.join(p,fn)
                cap = cv2.VideoCapture(oldpath)
                VideoWrite = cv2.VideoWriter(newpath,
                cv2.VideoWriter_fourcc(*'PIM1'),25, (340, 256))
                s = True
                while s:
                    s,frame = cap.read()
                    if s:
                        VideoWrite.write(cv2.flip(frame,1))
                VideoWrite.release()
                cap.release()
                num = int(fn[fn.find('a')+1:fn.find('.')])-1
                if p[-1] == 'n':
                    f1.write(newfn+" "+str(num)+"\n")
                else:
                    f2.write(newfn+" "+str(num)+"\n")
                print(newfn)
    cv2.destroyAllWindows()
    f1.close()
    f2.close()



def grayGen(path1, output):
    # tp = output
    # vp = output.replace('train','val')
    for p, d, f in os.walk(path1):
        for fn in f:
            if fn.endswith('.avi') and int(p[-2:])>9:
                newName = p[-3:]+fn
                oldpath = os.path.join(p,fn)
                cap = cv2.VideoCapture(oldpath)
                VideoWrite = cv2.VideoWriter(os.path.join(output, newName)
                , cv2.VideoWriter_fourcc(*'PIM1'),25, (340, 256))
                success,_ = cap.read()
                while success:
                    success,frame = cap.read()
                    try:
                        VideoWrite.write(frame[2:258,3:343,:])
                    except:
                        break
                cap.release()
                VideoWrite.release()
                print(newName,":completed")

    file1 = output.replace('train','eventTrain.txt')
    with open(file1,'a+') as f:
        f.truncate(0)
    f1 = open(file1,'a')
    for p,d,f in os.walk(output):
        for fn in f:
            num = int(fn[fn.find('a')+1:fn.find('.')])-1
            f1.write(fn+" "+str(num)+"\n")
    f1.close()

if __name__ == "__main__":
    path1 = "D:\\22_23Term1\\IEMS5910\\data"
    out = "D:\\22_23Term1\\IEMS5910\\rgbMini\\val"
    # traverse(path1,14,16,out)
    path1 = "E:\\DV_proj\\myenv\\label"
    out = "E:\\DV_proj\\myenv\\eventDataset\\train"
    # eventTra(path1, out)
    out = "E:\\DV_proj\\myenv\\grayDataset\\train"
    # grayGen(path1, out)
    eventFlip("E:\\DV_proj\\myenv\\eventDataset")
    