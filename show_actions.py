import cv2
import scipy.io

videopath = '/home/mostafa/git_workspace/Action-detection-in-shopping-dataset/Videos_MERL_Shopping_Dataset/Videos_MERL_Shopping_Dataset/1_1_crop.mp4'
mat = scipy.io.loadmat('1_1_label.mat')
labels = mat['tlabs']
frames2labels = {}

idx_to_action = {0:'Reach To Shelf',
                 1:'Retract From Shelf',
                 2:'Hand In Shelf',
                 3:'Inspect Product',
                 4:'Inspect Shelf' }
cap = cv2.VideoCapture(videopath)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for frame in range(1,count+1):
    frames2labels[frame]='None'

for i,action in enumerate(labels):
    for n in range(len(labels[i][0])):
        start = labels[i][0][n][0]
        end = labels[i][0][n][1]
        for key in range(start,end+1):
            frames2labels[key]=idx_to_action[i]

vid = cv2.VideoCapture(videopath)
cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))
frame_idx=0


while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frame_idx +=1
    cv2.putText(frame, frames2labels[frame_idx], (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4 ,lineType=cv2.LINE_AA)
    cv2.imshow('Stream', frame)
    ch = 0xFF & cv2.waitKey(20)
    if ch == 27:
        break
cv2.destroyAllWindows()



