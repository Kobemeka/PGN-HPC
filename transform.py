import numpy as np

def xrotation(rotation_angle):
    return np.array([
                [1, 0, 0],
                [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                [0, np.sin(rotation_angle), np.cos(rotation_angle)]
            ])
def yrotation(rotation_angle):
    return np.array([
                [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                [0, 1, 0],
                [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
            ])

def zrotation(rotation_angle):
    return np.array([
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1]
            ])

def translation(x,y,z):
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [x,y,z,1],
    ])

def transform(x,y,z,r):
    return np.array([
        [np.cos(r), -np.sin(r),0,x],
        [np.sin(r), np.cos(r),0,y],
        [0,0,1,z],
        [0,0,0,1],
    ])
    
def xyzRotation(angle):
    return {
        "x": xrotation(angle),
        "y": yrotation(angle),
        "z": zrotation(angle),
    }