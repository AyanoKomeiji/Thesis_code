import cv2
import numpy as np


class lattice:
    def __init__(self,img,center,ID,flame,color):
        self.img=img
        self.center=center
        self.ID=ID
        self.flame=flame
        self.color=color

def Remap_img(img,mtx,dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst

def read(fn, num, tag):
    im = cv2.imread(fn + str(num) + ".jpg")
    #mtx=np.loadtxt("F:\zemi\Structure from Motion\data\calibration\K_"+tag+".csv",delimiter=",")
    #dist=np.loadtxt("F:\zemi\Structure from Motion\data\calibration\d_"+tag+".csv",delimiter=",")
    #Remap_img(im,mtx,dist)
    img = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    #cv2.imwrite("F:\zemi\Structure from Motion\data\calibration\Remap_"+tag+"\img_"+str(num)+".jpg",img)
    return img


def make_lattice(img, left_top, right_down, size, flame):
    lattice_list = []
    ID=1
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for w in range(left_top[0], right_down[0], size):
        for h in range(left_top[1], right_down[1], size):
            center = (w + int(size / 2), h + int(size / 2))
            clip_img = gray[h:h + size, w:w + size]
            color = np.array([img[center[1]][center[0]][2],img[center[1]][center[0]][1],img[center[1]][center[0]][0]])
            new_lattice = lattice(clip_img, center, ID, flame, color)
            lattice_list.append(new_lattice)
            ID += 1
    return lattice_list


def draw_lattice(im, lattice_list, size, ofn, num):
    f = 0
    for a in lattice_list:
        center = a.center
        tl = (center[0] - int(size / 2), center[1] - int(size / 2))
        dr = (center[0] + int(size / 2), center[1] + int(size / 2))
        cv2.rectangle(im, tl, dr, (255 / len(lattice_list) * f, 0, 255 / len(lattice_list) * (len(lattice_list)-f)))
        f += 1
    cv2.imwrite(ofn + "img_" + str(num) + ".jpg", im)


def latMatch(im, laL, size1, size2, flame, Flag):
    lat_list = []
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    for lat in laL:
        img = gray.copy()
        temp = lat.img
        q_im = img[lat.center[1] - int(size2 / 2) :lat.center[1] + int(size2 / 2)+1,
               lat.center[0] - int(size2 / 2):lat.center[0] + int(size2 / 2)+1]
        res = cv2.matchTemplate(q_im, temp, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        new_center = (lat.center[0] + max_loc[0] - int(size2 / 2) + int(size1/2),
                      lat.center[1] + max_loc[1] - int(size2 / 2) + int(size1/2))
        if Flag == True:
            if max_val>0.7 and new_center[0]<lat.center[0]:
                clip_img =gray[new_center[1] - int(size1 / 2):new_center[1] + int(size1 / 2)+1,
                           new_center[0] - int(size1 / 2):new_center[0] + int(size1 / 2)+1]
                color = im[new_center[1]][new_center[0]]
                new_lattice = lattice(clip_img, new_center, lat.ID, flame, color)
                lat_list.append(new_lattice)
        else:
            if max_val>0.7 and new_center[0]>lat.center[0]:
                clip_img =gray[new_center[1] - int(size1 / 2):new_center[1] + int(size1 / 2)+1,
                           new_center[0] - int(size1 / 2):new_center[0] + int(size1 / 2)+1]
                color = im[new_center[1]][new_center[0]]
                new_lattice = lattice(clip_img, new_center, lat.ID, flame, color)
                lat_list.append(new_lattice)
    return lat_list


def track_lattice(fn, ofn, start, stop1, stop2, step, size_temp, size_find,left_top,width,height,tag):
    all_list = []
    for num in range(start,stop1,step):
        if num == start:
            img_start = read(fn,num,tag)
            laL1=make_lattice(img_start,left_top, (left_top[0]+width*size_temp, left_top[1]+height*size_temp), size_temp, num)
            all_list.append(laL1)
            #draw_lattice(img_start, laL1, size_temp, ofn, num)
        else:
            img_after = read(fn,num,tag)
            laL2 = latMatch(img_after, laL1, size_temp, size_find, num, True)
            all_list.append(laL2)
            laL1 = laL2
            #draw_lattice(img_after, laL2, size_temp, ofn, num)
    for num in reversed(range(stop2, start, step)):
        if num == start-step:
            laL1 = all_list[0]
        img_after = read(fn, num,tag)
        laL3 = latMatch(img_after, laL1, size_temp, size_find, num, False)
        all_list.append(laL3)
        laL1 = laL3
        #draw_lattice(img_after, laL3, size_temp, ofn, num)
    return all_list



'''
fn = "data/images/middle_1203/img_"
all_list = track_lattice(fn,30000,30090,5,11,17)
'''