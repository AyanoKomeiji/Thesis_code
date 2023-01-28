import numpy as np
import calibration as cal
import latticeMatch as lat

#im_p=[[u,v,theta],[u,v,theta], ... ]


tD_long = np.array([3.1,-3.1,18.2])
tD_middle = np.array([-5.1,-5.1, 14.8])
tD_short = np.array([-4.9,2.1, 12.3])
tD_upCam=[tD_middle,tD_long,tD_short]
tD_midCam=[tD_long, tD_middle, tD_short]
tD_lowCam=[tD_long, tD_middle, tD_short]
topData_up = np.loadtxt("data/reconstruct/CSV/tops_1203.csv", delimiter=",", dtype=float)
topData_mid = np.loadtxt("data/reconstruct/CSV/tops_1203_middle.csv", delimiter=",", dtype=float)
topData_low = np.loadtxt("data/reconstruct/CSV/tops_1203_lower.csv", delimiter=",", dtype=float)

points_up = cal.make_points(topData_up)
points_mid = cal.make_points(topData_mid)
points_low = cal.make_points(topData_low)

start_up = points_up[0][3]
start_mid = points_mid[0][3]
start_low = points_low[0][3]

P_up = cal.makeP(tD_upCam,points_up,start_up)
y_up = cal.makey(points_up)
C_up = cal.calC(P_up,y_up)
P_mid = cal.makeP(tD_midCam,points_mid,start_mid)
y_mid = cal.makey(points_mid)
C_mid = cal.calC(P_mid,y_mid)
P_low = cal.makeP(tD_lowCam,points_low,start_low)
y_low = cal.makey(points_low)
C_low = cal.calC(P_low,y_low)

C=np.concatenate([C_up,C_mid,C_low])
np.savetxt("data/reconstruct/CSV/C_up.csv",C_up, delimiter=",")
np.savetxt("data/reconstruct/CSV/C_mid.csv",C_mid, delimiter=",")
np.savetxt("data/reconstruct/CSV/C_low.csv",C_low, delimiter=",")
'''
#コーナー点の再構築
corner_all = np.loadtxt("data/reconstruct/corners_upper_1203.csv", delimiter=",", dtype=float)
reconst_corner = cal.reconst_for_corner(C_mid,corner_all,60000)
cal.out_ply(reconst_corner, "data/reconstruct/corner_upper.ply")

#竹串頂点の再構築
imp_long,imp_mid,imp_short=cal.make_imP(points_up,points_mid,points_low)
im_p=[imp_long,imp_mid,imp_short]
start=[start_up,start_mid,start_low]
Cam=[C_up,C_mid,C_low]
long_3d=cal.reconst_3d(Cam,imp_long,start)
middle_3d=cal.reconst_3d(Cam,imp_mid,start)
short_3d=cal.reconst_3d(Cam,imp_short,start)
print(long_3d,middle_3d,short_3d)
for points in im_p:
    td_list=[]
    for p,C,s in zip(points,Cam,start):
        re_3d=cal.reconst_3d_new(C,p,s)
        td_list.append(re_3d)
    print(td_list)

for points in im_p:
    td_list=[]
    for p,C,s in zip(points,Cam,start):
        re_3d=cal.reconst_3d_new(C,p[400:800],s-600)
        td_list.append(re_3d)
    print(td_list)
    
fn = "data/images/middle_1203/img_"
ofn1="data/lattice/middle_1203/1_"
ofn2="data/lattice/middle_1203/2_"
all_list1 = lat.track_lattice(fn, ofn1, 30000, 30310, 29700, 10, 7, 15)
all_list2 = lat.track_lattice(fn, ofn1, 30300, 30610, 30000, 10, 7, 15)
reconst_p2 = cal.reconstruct_for_lat(all_list2,C_mid,30000,(0,0,255))
reconst_p1 = cal.reconstruct_for_lat(all_list1,C_mid,30000,(0,255,0))
reconst_p = np.concatenate([reconst_p1, reconst_p2])
cal.out_ply(reconst_p, "data/reconstruct/middle_reconst2.ply")


C_up=np.loadtxt("data/reconstruct/CSV/C_up.csv", delimiter=",")
C_mid=np.loadtxt("data/reconstruct/CSV/C_mid.csv", delimiter=",")
C_low=np.loadtxt("data/reconstruct/CSV/C_low.csv", delimiter=",")
'''
#仏像の再構築
fn = "data/images/middle_1203/img_"
ofn1="data/lattice/mid_test/H_"
a=-104
fund=30000+a
flag1=30900+a
flag2=32100+a
start = 30300+a
stop = 30600+a
color=(255,0,0)
point1=(580,590)
w=24
h=110
C=C_mid
camera="B"
for num in range(start,stop,300):
    if num==start:
        all_list = lat.track_lattice(fn, ofn1, num, num + 310, num - 300, 10, 7, 15,point1,w,h,camera)
        reconst_all = cal.reconstruct_for_lat(all_list, C, fund, color)
    elif num>=flag1 and num<=flag2:
        all_list = lat.track_lattice(fn, ofn1, num, num + 310, num - 300, 10, 7, 19, point1, w, h,camera)
        reconst = cal.reconstruct_for_lat(all_list, C, fund, color)
        reconst_all = np.concatenate([reconst_all, reconst])
    else:
        all_list = lat.track_lattice(fn, ofn1, num, num + 310, num - 300, 10, 7, 15,point1,w,h,camera)
        reconst=cal.reconstruct_for_lat(all_list, C, fund, color)
        reconst_all=np.concatenate([reconst_all,reconst])

#print(reconst_all)
cal.out_ply(reconst_all, "data/reconstruct/middle_30300.ply")
