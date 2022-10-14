#!/usr/bin/env python
# Imports

import subprocess
import argparse
import os
import numpy as np
import main

# Parameters

parser = argparse.ArgumentParser(description='treats one frame using one set of parameters')
parser.add_argument('-pm', '--method', type=str, default='kmeans', help='Partitioning method - octree or kmeans')
parser.add_argument('-s', '--sequence', type=str, default='loot', help='Sequence to use - loot longdress soldier redandblack')
parser.add_argument('--frame', type=int, default=1, choices=range(1,301), metavar="[1-300]", help='Frame number from 1 to 300')
# octree
parser.add_argument('-bs', '--bsize', type=int, default=16, help='Size of the blocks after octree partitioning')
# kmeans
parser.add_argument('-cc', '--clustercounts', type=int, default=1500, help='Number of cluster to create using KMeans')
parser.add_argument('-l', '--lambd', type=float, default=0.3, help='Lambda parameter for colors')
parser.add_argument('-cs', '--colorspace', type=str, default='lab', choices=['yuv', 'y', 'rgb', 'lab'], help='Color space to use for kmeans cluster calculations')
parser.add_argument('-qsc', '--qstep_centers', type=float, default=10.0, help='Quantization step size for the centers')
parser.add_argument('-rm', '--ref_method', type=str, default='none', choices=['none', 'weight', 'weight1', 'VA'], help='Method used to refine the centers')
parser.add_argument('-ri', '--ref_iterations', type=int, default=10, help='Number of iteration for the refine centers algorithm')
parser.add_argument('-b', '--beta', type=float, default=2.0, help='Beta parameter for the generalized gaussian distribution used to refine centers with the "weight" method')
# miscellaneous
parser.add_argument('-d', '--dir', type=str, default="/tmp", help='Directory used to write bitstream files')
parser.add_argument('-dm','--decoder_match', action='store_true', help='Whether we check for encoder-decoder match')

args = parser.parse_args()
method = args.method
sequence = args.sequence
frame = args.frame
# octree
bsize = args.bsize
# kmeans
cc = args.clustercounts
lambd = args.lambd
colorspace = args.colorspace
qstep_centers = args.qstep_centers
ref_iterations = args.ref_iterations
ref_method = args.ref_method
beta = args.beta
# miscellaneous
bs_dir = args.dir
decoder_match_test = vars(args)['decoder_match']

qsteps = [1, 2, 4, 8, 12, 16, 20, 24, 32, 64]
# qsteps = [16, 32]
# qsteps = [16]

# Execute

results = main.run(method=method, sequence=sequence, frame_number=frame, qsteps=qsteps, bsize=bsize, clusters_count=cc, lambd=lambd, colorspace=colorspace, qstep_centers=qstep_centers, ref_iterations=ref_iterations, ref_method=ref_method, beta=beta, bitstream_directory=bs_dir, decoder_match_test=decoder_match_test)

# Results

results_dir = f"/path/to/results/{sequence}/{frame}"
if method == 'octree':
    results_filename = f'{method}_bsize{bsize}.txt'
elif method == 'kmeans':
    results_filename = f'{method}_cc-{cc}_lam-{lambd}_cs-{colorspace}_refmet-{ref_method}_refit-{ref_iterations}_qsc-{qstep_centers}.txt'

results_path = os.path.join(results_dir, results_filename)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

if results.shape[0] > 0:
    np.savetxt(results_path, results, fmt="%7d %2.8f %10d %10d %10d %10d %10d %10d %10.2f %10.2f %10.2f", header=f"{'qstep':>5} {'PSNR_Y':>11} {'bs_total':>10} {'bs_coeffs':>10} {'bs_centers':>10} {'bs_dupes':>10} {'dupes_count':>10} {'N_voxels':>10} {'time_part':>10} {'time_gft':>10} {'time_center_ref':>10}")
