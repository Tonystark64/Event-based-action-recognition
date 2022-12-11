import pandas as pd
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

def playRGBevent(path1, path2, adder=100):
    # path1 is rgb frame
    # path2 is event frame
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)
    for _ in range(10):
        cap1.read()
        cap2.read()
    matShift = np.float32([[1,0,3],[0,1,2]])
    # while s1 and s2:
    #     s1,f1 = cap1.read()
    #     s2,f2 = cap2.read()
    #     if s1 and s2:
    #         f2[f2<127] = 0
    #         f2[f2>=127] = adder
    #         # f2 = cv2.warpAffine(f2,matShift,(340,256))
    #         cv2.imshow('out',f1+f2)
    #     if cv2.waitKey(10)==27:
    #         break
    _, f1 = cap1.read()
    _, f2 = cap2.read()
    f2[f2<127] = 0
    f3 = f2.copy()
    f4 = f2.copy()
    f5 = f2.copy()
    f6 = f2.copy()
    f2[f2>=127] = adder
    f3[f3>=127] = 10
    f4[f4>=127] = 25
    f5[f5>=127] = 50
    f6[f6>=127] = 75
    plt.subplot(231);plt.imshow(f1,cmap='gray');plt.title('gray')
    plt.subplot(232);plt.imshow(f1+f3,cmap='gray');plt.title('add 10')
    plt.subplot(233);plt.imshow(f1+f4,cmap='gray');plt.title('add 25')
    plt.subplot(234);plt.imshow(f1+f5,cmap='gray');plt.title('add 50')
    plt.subplot(235);plt.imshow(f1+f6,cmap='gray');plt.title('add 75')
    plt.subplot(236);plt.imshow(f1+f2,cmap='gray');plt.title('add 100')
    plt.show()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def grayEvent(path1):
    path2 = path1.replace('gray', 'event')
    for p, d, f in os.walk(path1):
        for fn in f:
            if fn.endswith('.avi'):
                v1 = os.path.join(p,fn)
                v2 = v1.replace('gray','event')
                v3 = v1.replace('Dataset','Event20')
                VideoWrite = cv2.VideoWriter(v3,cv2.VideoWriter_fourcc(*'PIM1'),25,(340, 256))
                cap1 = cv2.VideoCapture(v1)
                cap2 = cv2.VideoCapture(v2)
                s1 = True
                s2 = True
                while s1 and s2:
                    s1,f1 = cap1.read()
                    s2,f2 = cap2.read()
                    if s1 and s2:
                        f2[f2<127] = 0
                        f2[f2>=127] = 20
                        # cv2.imshow('out',f1+f2)
                        VideoWrite.write(f1+f2)
                VideoWrite.release()
                print(v3,' added')
                cap1.release()
                cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path1 = "E:\\DV_proj\\myenv\\grayDataset"
    # grayEvent(path1)
    p1 = "E:\\DV_proj\\myenv\\grayDataset\\train\\V12a4.avi"
    p1 = "D:\\22_23Term1\\IEMS5910\\rgbDataset\\train\\v12a04.avi"
    p2 = "E:\\DV_proj\\myenv\\eventDataset\\train\\V12a19.avi"
    # playRGBevent(p1,p2)
    

    