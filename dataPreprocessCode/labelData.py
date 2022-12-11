import cv2
from dv import AedatFile
import pickle
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
import mediapipe as mp
import time
from matplotlib import pyplot as plt

# test for volunteer 08 action a07 waving arms
# about 15X play when traverse frames
# pickle.dump(data,open(output_path,'wb'))
# data = pickle.load(open(path,'rb'))
# event np.hstack()
# fps of black and white frames is 25, 0.04s between frames
# about 0.4--0.6 second to fill (346, 260) space
# f['events'].numpy() about 0.01s
# datetime.fromtimestamp(timestamp) unit:second
# timestamp = datetime.timestamp()

def playSegment(path,start_min,start_sec,end_min,end_sec,output_path = "",save=False):
    with AedatFile(path) as f:
        for frame in f['frames']:
            start_ts = frame.timestamp
            break
        if save:
            VideoWrite = cv2.VideoWriter(output_path, 
                cv2.VideoWriter_fourcc(*'PIM1'),25, (346, 260), isColor=False)
            cur = datetime(2022,10,27,0,0,0)
        start_a = start_ts + (start_min*60+start_sec)*1e6
        end_a = start_ts + (end_min*60+end_sec)*1e6
        for frame in f['frames']:
            if frame.timestamp < start_a:
                continue
            if frame.timestamp > end_a:
                break
            if save==False:
                cv2.imshow('out',frame.image)
            if save:
                VideoWrite.write(frame.image)
                print("current frame:{}".format(cur.time()))
                cur += timedelta(milliseconds=40)
            if save==False:
                if cv2.waitKey(10)==27:
                    break
        if save:
            VideoWrite.release()
        if save==False:
            cv2.destroyAllWindows()

def eventSegment(path,start_min,start_sec,end_min,end_sec):
    # event np.hstack()
    with AedatFile(path) as f:
        for frame in f['frames']:
            start_ts = frame.timestamp
            break
        start_a = start_ts + (start_min*60+start_sec)*1e6
        end_a = start_ts + (end_min*60+end_sec)*1e6
        res = []
        for e in f['events'].numpy():
            if e['timestamp'][-1] < start_a:
                continue
            if e['timestamp'][0] > end_a:
                break
            res.append(e)
        # pickle.dump(res,open('E:\\DV_proj\\myenv\\a8.pkl','wb'))

def extractTrain(path1,path2,path3):
    df = pd.read_csv(path2)
    # start_ts = int(df['timestamp'][0])
    with AedatFile(path1) as f:
        for frame in f['frames']:
            start_ts = int(frame.timestamp)
            break
    counter = 1
    n_a = df.shape[0]
    VideoWrite = cv2.VideoWriter(path3+"\\a"+str(counter)+".avi", 
        cv2.VideoWriter_fourcc(*'PIM1'),25, (346, 260), isColor=False)
    print("Saving %d action frame..."%(counter))
    start_a = start_ts + (df['min_start'][counter-1]*60+df['sec_start'][counter-1])*1e6
    end_a = start_ts + (df['min_end'][counter-1]*60+df['sec_end'][counter-1])*1e6
    with AedatFile(path1) as f:
        for frame in f['frames']:
            if frame.timestamp < start_a:
                continue
            if frame.timestamp > end_a:
                VideoWrite.release()
                if counter >= n_a:
                    break
                counter += 1
                start_a = start_ts + (df['min_start'][counter-1]*60+df['sec_start'][counter-1])*1e6
                end_a = start_ts + (df['min_end'][counter-1]*60+df['sec_end'][counter-1])*1e6
                VideoWrite = cv2.VideoWriter(path3+"\\a"+str(counter)+".avi", 
                    cv2.VideoWriter_fourcc(*'PIM1'),25, (346, 260), isColor=False)
                print("Saving %d action frame..."%(counter))
            if frame.timestamp <= end_a:
                VideoWrite.write(frame.image)

        print("Extract frames completed...")

        res = []
        counter = 1
        print("Saving %d action event..."%(counter))
        start_a = start_ts + (df['min_start'][counter-1]*60+df['sec_start'][counter-1])*1e6
        end_a = start_ts + (df['min_end'][counter-1]*60+df['sec_end'][counter-1])*1e6
        for e in f['events'].numpy():
            if e['timestamp'][-1] < start_a:
                continue
            if e['timestamp'][0] > end_a:
                pickle.dump(res,open(path3+"\\a"+str(counter)+".pkl",'wb'))
                if counter >= n_a:
                    break
                res = []
                counter += 1
                print("Saving %d action event..."%(counter))
                start_a = start_ts + (df['min_start'][counter-1]*60+df['sec_start'][counter-1])*1e6
                end_a = start_ts + (df['min_end'][counter-1]*60+df['sec_end'][counter-1])*1e6
            else:
                res.append(e)
                
def eventOverlap(path1, path2, path3, action, count=100):
    # path1 is frame segment
    # path2 is event segment
    # path3 is csv file
    df = pd.read_csv(path3)
    start_ts = int(df['timestamp'][0])
    action = action - 1
    count = count
    start_a = start_ts + (df['min_start'][action]*60+df['sec_start'][action])*(1e6)
    # 40000 microsecond is one frame
    # jump first 100 frame
    start_a += 40000*count
    end_a = start_a + 40000
    e = pickle.load(open(path2,'rb'))
    e = np.hstack(e)
    e = e[e['timestamp'] <= end_a]
    e = e[e['timestamp'] >= start_a]
    e = e[['x','y','polarity']]
    cap = cv2.VideoCapture(path1)
    for _ in range(count+2):
        _,frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    adder = 1
    plt.subplot(231);plt.imshow(frame,cmap='gray');plt.title('original')
    frame1 = frame.copy()
    frame1[:][:] = 0
    frame2 = frame1.copy()
    frame3 = frame1.copy()
    frame4 = frame1.copy()
    frame5 = frame1.copy()
    for i in e:
        frame1[i['y']][i['x']] += adder
    for i in e:
        if i['polarity']:
            frame2[i['y']][i['x']] += adder
            frame3[i['y']][i['x']] += adder
            frame5[i['y']][i['x']] -= adder
        else:
            frame2[i['y']][i['x']] -= adder
            frame4[i['y']][i['x']] += adder
            frame5[i['y']][i['x']] += adder
    # print(np.unique(frame2))
    plt.subplot(232);plt.imshow(frame1,cmap='gray');plt.title('pos/neg add %d'%(adder))
    plt.subplot(233);plt.imshow(frame2,cmap='gray');plt.title('pos add %d,neg minus %d'%(adder,adder))
    plt.subplot(234);plt.imshow(frame5,cmap='gray');plt.title('pos minus %d,neg add %d'%(adder,adder))
    plt.subplot(235);plt.imshow(frame3,cmap='gray');plt.title('only pos')
    plt.subplot(236);plt.imshow(frame4,cmap='gray');plt.title('only neg')
    plt.show()

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
    

if __name__ == "__main__":
    path1 = "E:\\DV_proj\\myenv\\dvSave-2021_07_23_15_04_32_S013T001.aedat4"
    path2 = "E:\\DV_proj\\myenv\\label\\V13\\V13.csv"
    path3 = "E:\\DV_proj\\myenv\\label\\V13"

    extractTrain(path1,path2,path3)

    # playSegment(path1,0,0,100,0,"E:\\DV_proj\\myenv\\sampleV13.avi",True)


    video_p = "E:\\DV_proj\\myenv\\label\\V08\\a7.avi"
    event_p = "E:\\DV_proj\\myenv\\label\\V06\\a12.pkl"
    csv_p = "E:\\DV_proj\\myenv\\label\\V08\\V08.csv"

    # eventOverlap(video_p,event_p,csv_p, 7)

    path4 = "E:\\DV_proj\\myenv\\eventDataset\\train"
    # genEventFrame(event_p, path4)
