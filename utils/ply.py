# Parts of this source file are adapted from https://github.com/STAC-USC/RA-GFT

import numpy as np
import os

def read(filename):
    N = -1  # number of points
    w = -1  # width

    # Parse header
    with open(filename, "r") as f:
        for line_num, line in enumerate(f):  # read lines until header ends
            if line.startswith('end_header'):  # end of header
                header_line_num = line_num
                break
            elif line.startswith('element vertex'):  # read number of elements
                N = int(line.rstrip().split(' ')[-1])
            elif line.startswith('comment width'):  # read width
                w = int(line.rstrip().split(' ')[-1])

    # Calculate bit resolution
    if w != -1:
        J = np.log2(w + 1)

    # Read in data
    tmp = np.genfromtxt(
        fname=filename, dtype=np.float32, skip_header=header_line_num+1)

    # Split into vertices and attributes
    V = tmp[:, 0:3]
    A = tmp[:, 3:6].astype(np.uint8)

    V = V.astype(np.int32)

    return V, A, N, J

def write(V, A, path):

    VA = np.concatenate((V, A), axis=1)
    
    header = f"""ply
format ascii 1.0
comment Version 2, Copyright 2017, 8i Labs, Inc.
comment frame_to_world_scale 0.181985
comment frame_to_world_translation -31.8478 1.0016 -32.6788
comment width 1023
element vertex {len(V)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header"""
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, VA, fmt='%d', header=header, comments="")
    

def export_pc_color_blocks(V ,idx_start, idx_stop, path):

    A = np.zeros((V.shape[0], 3), dtype=np.uint32)

    for it in range(len(idx_start)):
        random_color = np.random.choice(range(256), size=(1,3))
        block_length = idx_stop[it] - idx_start[it]
        Vb = V[idx_start[it]:idx_stop[it], :]
        A[idx_start[it]:idx_stop[it], :] = np.repeat(random_color, block_length, axis=0)
        
    write(V, A, path)
