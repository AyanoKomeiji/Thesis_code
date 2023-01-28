import numpy as np
import numpy.linalg as LIA
import math
from matplotlib import pyplot as plt
import cv2


class same_points:
    def __init__(self,ID,color):
        self.list=[]
        self.ID=ID
        self.color=color

    def append(self,points):
        self.list.append(points)


def make_points(topData):
    points1 = []
    points2 = []
    points3 = []
    for top in topData:
        if int(top[2]) == 1:
            points1.append([top[0], -top[1], top[2], top[3]])
        elif int(top[2]) == 2:
            points2.append([top[0], -top[1], top[2], top[3]])
        elif int(top[2]) == 3:
            points3.append([top[0], -top[1], top[2], top[3]])
    points = np.concatenate([points1, points2, points3])
    return points


def makey(points):
    y = []
    leng=len(points)
    if leng != 3:
        for p in points:
            u = p[0]
            v = p[1]
            y.append([u, v])
    else:
        for num in range(leng):
            for p in points[num]:
                u = p[0]
                v = p[1]
                y.append([u, v])
    y = np.reshape(y, [len(y) * 2, 1], order="F")
    return y


def makeP(tD, top, start):
    Pu = []
    Pv = []
    for n in top:
        if n[2] == 1.0:
            tD_point = tD[0]
        elif n[2] == 2.0:
            tD_point = tD[1]
        else:
            tD_point = tD[2]
        theta = math.radians(-(n[3] - start) * 360 / 3663)
        X = tD_point[0] * math.cos(theta) - tD_point[1] * math.sin(theta)
        Y = tD_point[0] * math.sin(theta) + tD_point[1] * math.cos(theta)
        Z = tD_point[2]
        u = n[0]
        v = n[1]
        Un = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z]
        Pu.append(Un)
        Vn = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z]
        Pv.append(Vn)
    Pu = np.array(Pu)
    Pv = np.array(Pv)
    P = np.concatenate([Pu, Pv])
    return P


def calC(P, y):
    P1 = P.T @ P
    C = LIA.pinv(P1) @ P.T @ y
    return C


def make_imP(up_p, mid_p, low_p):
    long_up=[]
    middle_up=[]
    short_up=[]
    long_mid=[]
    middle_mid=[]
    short_mid=[]
    long_low=[]
    middle_low=[]
    short_low=[]
    for p in up_p:
        if int(p[2])==1:
            middle_up.append(p)
        elif int(p[2])==2:
            long_up.append(p)
        else:
            short_up.append(p)
    for p in mid_p:
        if int(p[2])==1:
            long_mid.append(p)
        elif int(p[2])==2:
            middle_mid.append(p)
        else:
            short_mid.append(p)
    for p in low_p:
        if int(p[2])==1:
            long_low.append(p)
        elif int(p[2])==2:
            middle_low.append(p)
        else:
            short_low.append(p)
    long_p =[np.array(long_up), np.array(long_mid), np.array(long_low)]
    middle_p = [np.array(middle_up), np.array(middle_mid), np.array(middle_low)]
    short_p = [np.array(short_up), np.array(short_mid), np.array(short_low)]
    return long_p,middle_p,short_p


def reconst_3d(Cam, im_p, start):
    U = []
    V = []
    Yu = []
    Yv = []
    for num in range(len(Cam)):
        C=Cam[num].T[0]
        for p in im_p[num]:
            theta = math.radians(-(p[3] - start[num]) * 360 / 3663)
            cos = math.cos(theta)
            sin = math.sin(theta)
            u = p[0]
            v = p[1]
            fu = C[0] * cos + C[1] * sin - C[8] * u * cos - C[9] * u * sin
            gu = -C[0] * sin + C[1] * cos + C[8] * u * sin - C[9] * u * cos
            hu = C[2] - C[10] * u
            U.append([fu, gu, hu])
            fv = C[4] * cos + C[5] * sin - C[8] * v * cos - C[9] * v * sin
            gv = -C[4] * sin + C[5] * cos + C[8] * v * sin - C[9] * v * cos
            hv = C[6] - C[10] * v
            V.append([fv, gv, hv])
            u = p[0]-C[3]
            v = p[1]-C[7]
            Yu.append(u)
            Yv.append(v)
    U = np.array(U)
    V = np.array(V)
    Yu = np.array(Yu)
    Yv = np.array(Yv)
    F = np.concatenate([U, V])
    y= np.concatenate([Yu,Yv])
    point_3D = calC(F, y)
    return point_3D


def reconst_3d_new(C, im_p, start):
    U = []
    V = []
    y = []
    C=C.T[0]
    for p in im_p:
        theta = math.radians(-(p[3] - start) * 360 / 3663)
        cos = math.cos(theta)
        sin = math.sin(theta)
        u = p[0]
        v = p[1]
        fu = C[0] * cos + C[1] * sin - C[8] * u * cos - C[9] * u * sin
        gu = -C[0] * sin + C[1] * cos + C[8] * u * sin - C[9] * u * cos
        hu = C[2] - C[10] * u
        U.append([fu, gu, hu])
        fv = C[4] * cos + C[5] * sin - C[8] * v * cos - C[9] * v * sin
        gv = -C[4] * sin + C[5] * cos + C[8] * v * sin - C[9] * v * cos
        hv = C[6] - C[10] * v
        V.append([fv, gv, hv])
        u = p[0]-C[3]
        v = p[1]-C[7]
        y.append([u,v])
    y = np.reshape(y, [len(y) * 2, 1], order="F")
    U = np.array(U)
    V = np.array(V)
    F = np.concatenate([U, V])
    point_3D = calC(F, y)
    return point_3D


def reconstruct_for_lat(lat_list, C, start, color):
    reconst_points = []
    lat_lists = []
    for lat in lat_list[0]:
        exec("new_points"+str(lat.ID)+"=same_points(lat.ID,lat.color)")
        exec("new_points" + str(lat.ID) + ".append([lat.center[0]-540,-lat.center[1] +960, lat.ID,lat.flame])")
        exec("lat_lists.append(new_points"+str(lat.ID)+")")
    for flame in range(1, len(lat_list)):
        for lat in lat_list[flame]:
            exec("new_points" + str(lat.ID) + ".append([lat.center[0] - 540,-lat.center[1] + 960, lat.ID, lat.flame])")
    for points in lat_lists:
        if len(points.list) > 45:
            tD_P = reconst_3d_new(C, points.list, start)
            if color==(0,0,0):
                reconst_points.append([tD_P.T[0][0],tD_P.T[0][1],tD_P.T[0][2],points.color[0],points.color[1],points.color[2]])
            else:
                reconst_points.append([tD_P.T[0][0], tD_P.T[0][1], tD_P.T[0][2], color[0], color[1], color[2]])
    reconst_points = np.array(reconst_points)
    return reconst_points

def reconst_for_corner(C, corners, start):
    corner_list=[]
    reconst_points=[]
    for corner in corners:
        if len(corner_list)<28:
            corner_list.append([[corner[0]-540,-corner[1]+960,corner[2],corner[3]]])
        else:
            corner_list[int(corner[2]%28)].append([corner[0]-540,-corner[1]+960,corner[2],corner[3]])
    for points in corner_list:
        tD_P = reconst_3d_new(C, points[:60], start)
        reconst_points.append([tD_P.T[0][0],tD_P.T[0][1],tD_P.T[0][2],255,255,255])
    reconst_points = np.array(reconst_points)
    return reconst_points


def out_ply(point_3d,ofn):
    with open(ofn,mode='w') as f:
        f.write("ply\nformat ascii 1.0\ncomment 3D reconstruct\nelement vertex \n"
                "property double x\nproperty double y\nproperty double z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point in point_3d:
            '''
            f.write("\n"+str(point[0]) + " " + str(point[1]) + " " + str(0) + " " + str(
                    int(point[4]))+ " " + str(int(point[5])) + " " + str(int(point[6])))
            '''
            f.write("\n"+str(point[0])+" "+str(point[1])+" "+str('{:.2f}'.format(point[2]))+" "+str(int(point[3]))
                    +" "+str(int(point[4]))+" "+str(int(point[5])))



def correlation_vector(a, b):
    a_norm = np.linalg.norm(a)  # aのL2ノルム
    b_norm = np.linalg.norm(b)  # bのL2ノルム
    r = np.dot(a.T, b) / (a_norm * b_norm)  # 相関係数
    return r


def estimateUV(topData, C, tD):
    u_list = []
    v_list = []
    def_list = []
    flame = []
    for num in range(len(topData)):
        flame.append(topData[num][2])
        theta = math.radians(-(topData[num][2] - 600) * 360 / 3663)
        cos = math.cos(theta)
        sin = math.sin(theta)
        x = tD[0] * cos - tD[1] * sin
        Y = tD[0] * sin + tD[1] * cos
        z = tD[2]
        UV = (C[8] * x + C[9] * Y + C[10] * z + 1)
        U1 = (C[0] * x + C[1] * Y + C[2] * z + C[3])
        V1 = (C[4] * x + C[5] * Y + C[6] * z + C[7])
        U = U1 / UV
        V = V1 / UV
        u_list.append(U[0])
        v_list.append(V[0])
        def_list.append([U[0], V[0]])
    def_list = np.array(def_list)
    return def_list, flame


def euclid_dis(def_list, topData, y):
    defO = y.reshape((len(topData), 2), order="F")
    dif = def_list - defO
    d_list = []
    for a in dif:
        d_list.append((a[0] ** 2 + a[1] ** 2) ** 0.5)

    return d_list

def judge_correct(tD, topData):
    X = np.array([0, 0, 0])
    a = 0
    dot = (X - tD) @ (X - tD)
    while dot > 0.0005:
        y = makey(topData)
        P = makeP(tD, topData)
        C = calC(P, y)
        est_uv, flame = estimateUV(topData, C, tD)
        d_list = euclid_dis(est_uv, topData, y)

        # plt.plot(flame, d_list)
        # plt.show()
        index = d_list.index(max(d_list))
        d_list.pop(index)

        topData = np.delete(topData, index, 0)
        X = reconst_3d(C, topData)[:3]
        X = X.reshape((1, 3))[0]
        dot = (X - tD) @ (X - tD)

    y = makey(topData)
    P = makeP(tD, topData)
    C = calC(P, y)
    est_uv, flame = estimateUV(topData, C, tD)
    d_list = euclid_dis(est_uv, topData, y)
    plt.plot(flame, d_list)
    plt.show()
    print(len(topData))
    return C, X


def draw_ellipse(points,fn,ofn):
    img=cv2.imread(fn)
    img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    for p in points:
        if int(p[2])==1:
            cv2.circle(img,(int(p[0]+img.shape[1]/2),int(-p[1]+img.shape[0]/2)),1,(0,0,255))
        elif int(p[2])==2:
            cv2.circle(img,(int(p[0]+img.shape[1]/2),int(-p[1]+img.shape[0]/2)),1,(0, 255, 0))
        elif int(p[2])==3:
            cv2.circle(img,(int(p[0]+img.shape[1]/2),int(-p[1]+img.shape[0]/2)),1, (255, 0, 0))
        else:
            cv2.circle(img,(int(p[0]+img.shape[1]/2),int(-p[1]+img.shape[0]/2)),1,(0, 0, 0))
    cv2.imwrite(ofn,img)


