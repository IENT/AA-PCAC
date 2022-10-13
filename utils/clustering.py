import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy.spatial.distance as distance
import scipy.stats as stats
from scipy.special import gamma

import utils.color as color
import utils.morton as morton
import utils.quantization as quant


def kmeans_encoder(V, A, A_rgb, partition_count, lam_part, colorspace='lab', qstep_centers=1, emulate_dec=True, random_state=0):
    # change the color space of A (which is assumed to be yuv)
    if colorspace == 'yuv':
        Ac = A
    elif colorspace == 'y':
        Ac = A[:,0:1]
    elif colorspace == 'rgb':
        Ac = A_rgb.astype('float64')
    elif colorspace == 'lab':
        Ac = color.rgb_to_lab(A_rgb)
    else:
        raise ValueError(f'Color space {colorspace} not understood for runing kmeans.')

    # scale points and attributes
    VA_scaled, scaler_VA = concatenate_data(V=V, A=Ac, lam_part=lam_part)

    # clustering
    kmeans = MiniBatchKMeans(
        n_clusters=partition_count,
        init='k-means++',          # clusters initialization
        random_state=random_state,
        verbose=False
    ).fit(X=VA_scaled)
    # get encoder labels (for evaluation only)
    labels_encoder = kmeans.predict(VA_scaled)
    
    # un-scale centers
    centers_encoder = scaler_VA.inverse_transform(kmeans.cluster_centers_)

    # prepare centers for decoder
    centers_decoder = centers_encoder.copy()
    # remove empty
    centers_decoder, partition_count = remove_empty_centers(
        centers=centers_decoder, labels=labels_encoder
    )
    # remove attribute dimensions, quantize and morton order (prepare for transmission)
    centers_decoder, partition_count = prepare_centers_transmission(
        centers=centers_decoder, dtype=V.dtype, qstep_centers=qstep_centers
    )
    # obtain labels at decoder
    labels_decoder = labels_from_centers(V=V, centers=centers_decoder)

    # get new block boundaries. V and A get reordered
    idx_start, idx_stop, N_block, V_ordered, A_ordered = block_indices(V, A, labels_decoder, partition_count)

    return centers_encoder, labels_encoder, centers_decoder, labels_decoder, V_ordered, A_ordered, idx_start, idx_stop, N_block
    
def concatenate_data(V, A, lam_part):
    # Concatenate geometry and attributes
    VA = np.concatenate((V, A), axis=1)

    # Scale to zero mean and unit variance
    scaler = preprocessing.StandardScaler().fit(VA)
    VA_scaled = scaler.transform(VA)
    VA_scaled[:, 3:] *= lam_part

    return VA_scaled, scaler


def prepare_centers_transmission(centers, dtype, qstep_centers=1):
    # only vertices
    centers = centers[:, 0:3]

    # quantize and remove duplicates
    centers, _ = quant.quantize(centers, qstep_centers)
    centers = np.unique(centers, axis=0)

    # adjust number of partitions
    partition_count = centers.shape[0]

    # morton order
    centers_int = centers.astype(np.uint32)
    m = morton.morton3D_64_encode(
        x=centers_int[:, 2], y=centers_int[:, 1], z=centers_int[:, 0])
    sort_morton_idx = np.argsort(m)
    centers = centers[sort_morton_idx, :]

    return centers, partition_count


def remove_empty_centers(centers, labels):
    # Detect empty partitions
    mask = np.ones(centers.shape[0], dtype=bool)
    for center_id in range(centers.shape[0]):
        if np.sum(labels == center_id) == 0:
            mask[center_id] = False

    # Remove centers
    centers = centers[mask, :]

    # Adjust number of partitions
    partition_count = centers.shape[0]

    # "Report" back
    # num_deleted = mask.size - num_partitions
    # if num_deleted > 0:
    #     print(f"{num_deleted} empty partitions deleted")

    return centers, partition_count


def labels_from_centers(V, centers, method='kmeans', random_state=0):
    # calculate labels given partition centers
    # call clustering method for a single step (E-step)
    if method == 'kmeans':
        kmeans = KMeans(
            n_clusters=centers.shape[0], init=centers, n_init=1, max_iter=1,
            random_state=random_state
        ).fit(X=V)
    elif method == 'batchkmeans':
        kmeans = MiniBatchKMeans(
            n_clusters=centers.shape[0], init=centers, n_init=1, max_iter=1,
            random_state=random_state
        ).fit(X=V)

    labels = kmeans.labels_

    return labels

def refine_centers(
        V, A_yuv, A_rgb, centers_init, labels_init, N_iter=5, method='weight', qstep_centers=1, beta=1, scaling=False,
        random_state=0, lam_part=0.2
):

    A = color.rgb_to_lab(A_rgb)

    scaler_V = scaler_A = scaler_VA = None
    if scaling and (method == 'weight' or method == 'weight1'):
        # Warning:
        # This is still buggy. I don't know why; scaling A is not problematic.
        # But scaling V results in worse performance on A
        # However, weights are normalized between [0, 1].
        # Where is the problem?!
        return  # Remove if you have read above statement and want to debug
        extra_scale = 100  # PDF function has problems with small values
        scaler_V = preprocessing.StandardScaler().fit(V)
        V_scaled = scaler_V.transform(V)*extra_scale
        centers_init = scaler_V.transform(centers_init)*extra_scale

        scaler_A = preprocessing.StandardScaler().fit(A)
        A_scaled = scaler_A.transform(A)*extra_scale
    else:
        V_scaled = V.copy()
        A_scaled = A.copy()

    if method == 'VA':
        VA_scaled, scaler_VA = concatenate_data(V=V, A=A, lam_part=lam_part)
        V_scaled = VA_scaled[:, 0:3]
        A_scaled = VA_scaled[:, 3:]
        centers_init = scaler_VA.transform(centers_init)
        centers_init[:, 3:] *= lam_part

    # Init
    num_partitions = centers_init.shape[0]
    dist = {  # distances
        'V': np.zeros(N_iter + 1),
        'A': np.zeros(N_iter + 1)
    }
    centers = centers_init.copy()
    labels = labels_init.copy()

    # Loop
    for it in range(N_iter):
        # Update centers
        centers_old = centers.copy()
        centers = np.zeros((centers_old.shape[0], centers_init.shape[1]))

        for cluster_id in range(centers_old.shape[0]):
            # Select points of each partition given by labels calculated on V
            mask = labels == cluster_id
            num_points = np.sum(mask)

            Vc = V_scaled[mask, :]
            Ac = A_scaled[mask, :]

            # Update distances (previous run)
            center_old_V = centers_old[cluster_id, 0:3]
            if centers_old.shape[1] == 3:
                center_old_A = np.mean(Ac, axis=0)
            else:
                center_old_A = centers_old[cluster_id, 3:]

            # distances need to be scaled back for VA method, in order to be comparable
            # with the distances produced by the other methods
            if method == 'VA':
                scaled_back = scaler_VA.inverse_transform(np.concatenate((Vc, Ac), axis=1))
                scaled_back_center = scaler_VA.inverse_transform(centers_old[cluster_id, np.newaxis])
                dist['V'][it] += calculate_distance(
                    X=scaled_back[:, :3],
                    center=scaled_back_center[:, :3]
                )/num_partitions
                dist['A'][it] += calculate_distance(
                    X=scaled_back[:, 3:],
                    center=scaled_back_center[:, 3:]
                )/num_partitions
            else:
                dist['V'][it] += calculate_distance( X=Vc, center=center_old_V )/num_partitions
                dist['A'][it] += calculate_distance( X=Ac, center=center_old_A )/num_partitions

            if method == 'weight' or method == 'weight1':
                if num_points > 1:  # Multiple points in partition
                    # Calculate weights
                    weights = np.ones(Ac.shape[0])
                    if method == 'weight':
                        # Mean vector and covariance of attributes
                        mean_A = np.mean(Ac, axis=0)
                        var_A = np.var(Ac, axis=0, ddof=1)
                        if np.abs(np.sum(var_A)) > 1e-6:
                            for dim in range(Ac.shape[1]):
                                alpha = np.sqrt(var_A[dim]*gamma(1/beta)/gamma(3/beta))  # sigma^2 = alpha^2 * gamma(3/beta) / gamma(1/beta)
                                weights *= stats.gennorm(beta=beta, loc=mean_A[dim], scale=alpha).pdf(Ac[:, dim])

                    # Calculate centers as weighted mean
                    centers[cluster_id, :] = np.sum(
                        (weights[:, np.newaxis] * Vc), axis=0
                    )/np.sum(weights)

                elif num_points > 0:  # Single point in partition
                    centers[cluster_id, :] = Vc
                else: # No point in partition
                    centers[cluster_id, :] = centers_old[cluster_id, :]

            elif method == 'VA':  # Calculate centers as mean
                centers[cluster_id, :] = np.mean(VA_scaled[mask, :], axis=0)

        # Update labels
        centers_V = centers[:, 0:3]
        labels = labels_from_centers(
            V=V_scaled, centers=centers_V, random_state=random_state)

    # Undo scaling
    if scaler_V is not None:
        centers = scaler_V.inverse_transform(centers/extra_scale)

    if scaler_VA is not None:
        centers = scaler_VA.inverse_transform(centers)
        centers = centers[:, 0:3]
        
    # Distances for final iteration
    dist['V'][it+1], dist['A'][it+1] = calculate_distances(
        V=V_scaled, A=A_scaled, centers=centers, labels=labels
    )

    # remove empty
    centers, partition_count = remove_empty_centers(
        centers=centers, labels=labels
    )
    # quantize and morton order (prepare for transmission)
    centers, partition_count = prepare_centers_transmission(
        centers=centers, dtype=V.dtype, qstep_centers=qstep_centers
    )
    # obtain labels at decoder
    labels = labels_from_centers(V=V, centers=centers)

    # get new block boundaries. V and A get reordered
    idx_start, idx_stop, N_block, V_ordered, A_ordered = block_indices(V, A_yuv, labels, partition_count)

    return centers, labels, V_ordered, A_ordered, idx_start, idx_stop, N_block, dist


def calculate_distance(X, center):
    return np.sum(
        # pairwise.euclidean_distances(X, np.atleast_2d(center)).ravel()
        distance.cdist(X, np.atleast_2d(center)).ravel()
    )


def calculate_distances(V, A, centers, labels):

    num_partitions = centers.shape[0]
    dist_V = np.zeros(num_partitions)
    dist_A = np.zeros(num_partitions)
    for cluster_id in range(num_partitions):

        mask = labels == cluster_id
        Vc = V[mask, :]
        Ac = A[mask, :]

        centers_V = centers[cluster_id, 0:3]
        if centers.shape[1] == 3:
            centers_A = np.mean(Ac, axis=0)
        else:
            centers_A = centers[cluster_id, 3:]

        dist_V[cluster_id] = calculate_distance(Vc, centers_V)
        dist_A[cluster_id] = calculate_distance(Ac, centers_A)

    return np.mean(dist_V), np.mean(dist_A)


def block_indices(V, A, labels, clusters_count):
    # block boundaries
    V_kmeans = np.zeros((0, 3), dtype=np.uint32)
    A_kmeans = np.zeros((0, 3), dtype=np.float32)
    idx_start = np.zeros((1), dtype=np.uint32)
    idx_stop = np.zeros((0), dtype=np.uint32)
    idx_accumulator = 0
    for cluster_id in range(clusters_count):
        current_vertices = V[ np.where(labels == cluster_id, True, False) ]
        current_attributes = A[ np.where(labels == cluster_id, True, False) ]
        V_kmeans = np.concatenate((V_kmeans, current_vertices))
        A_kmeans = np.concatenate((A_kmeans, current_attributes))
        idx_accumulator += len(current_vertices)
        idx_start = np.append(idx_start, idx_accumulator)
        idx_stop = np.append(idx_stop, idx_accumulator)
    idx_start = idx_start[:-1]
    # Number of points per block
    num_per_block = idx_stop - idx_start
    
    return idx_start, idx_stop, num_per_block, V_kmeans, A_kmeans
