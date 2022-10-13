# Parts of this source file are adapted from https://github.com/STAC-USC/RA-GFT

import numpy as np
from tqdm import tqdm

import utils.graph as graph


def compute_gft(W, Q=None):
    """ Computes GFT from weight matrix

    Args:
        W (np.array): weight matrix
        Q (np.array): node weight matrix

    Returns:
        np.array: GFT matrix (sorted eigen vectors of Laplacian)
        np.array: GFT frequency vector (sorted eigen values of Laplacian)
    """

    if W.shape[0] > 1:  # more than one point in graph

        if Q is None:
            Q = np.ones((W.shape[0], 1))

        Qm = np.diag(1/np.sqrt(Q.flatten()))  # assume Q is a vector
        L = graph.w2l(W)  # Compute Laplacian
        Ln = Qm @ L @ Qm  # normalize Laplacian

        # Eigenvalue decomposition
        # We use eigh instead of eig because it is much faster and
        # since Ln is symmetric
        diagD, GFT = np.linalg.eigh(Ln)  # eigenvalues, eigenvectors

        # Sort eigenvalues and vectors correspondingly
        idxSorted = np.argsort(diagD)
        Gfreq = diagD[idxSorted]
        GFT = GFT[:, idxSorted]

        # Ensure non-negative DC
        GFT[:, 0] = np.abs(GFT[:, 0])
        Gfreq[0] = np.abs(Gfreq[0])

        GFT = GFT.T  # Transpose

        # Compute GFT for unconnected graphs
        num_subgraphs_block = graph.num_subgraphs_from_gfreq(Gfreq)
        if num_subgraphs_block > 1:
            # Graph is subdivided into subgraphs
            # print('Found {} components'.format(num_subgraphs_block))
            GFT = []  # forget everything we did so far
            Gfreq = []

            # Extract subgraphs indices
            idx_components = graph.compute_subgraphs_ids(W)

            # Not connected, calculate everything again, yay
            for idx_component in idx_components:
                # Weight matrix and Q vector per subgraph
                Wc = W[np.ix_(idx_component, idx_component)]
                Qc = Q[idx_component]

                # Calculate GFT per subgraph
                GFTc, Gfreqc = compute_gft(W=Wc, Q=Qc)

                # Set entries of GFT matrix to zero for all other points not
                # belonging to subgraph
                U = np.zeros((W.shape[0], Wc.shape[0]))
                U[idx_component, :] = GFTc.T  # transpose

                # Save values per subgraph
                GFT.append(U)
                Gfreq.append(Gfreqc)

            GFT = np.hstack(GFT).T  # combine and revert transpose
            Gfreq = np.concatenate(Gfreq)

    else:  # 1D-case, DC only
        # TODO: Q is not needed here?!
        GFT = np.array([1.0])
        Gfreq = np.array([0.0])

    return GFT, Gfreq


def transform_gft(X, GFT):
    """ GFT transform

    Args:
        X (np.array): graph signal
        GFT (np.array): GFT matrix

    Returns:
        np.array: graph spectrum
    """
    if GFT.shape[0] > 1:
        Xhat = GFT @ X  # actual transform
    else:
        Xhat = X  # 1D case

    return Xhat


def itransform_gft(Xhat, GFT):
    """ Inverse GFT transform

    Args:
        Xhat (np.array): graph spectrum
        GFT (np.array): GFT matrix

    Returns:
        np.array: graph signal
    """
    if GFT.shape[0] > 1:
        X = GFT.T @ Xhat
    else:  # 1D case
        X = Xhat

    return X


def compute_graph_and_gft(V, A=None, Q=None, thresh=np.sqrt(3)+0.00001,
                          lam=0, res=np.array([]), res_method='idx',
                          dist_thresh=2):
    """ Construct graph and compute GFT

    Args:
        V (np.array): vertices
        Q (np.array, optional): node weight vector. Defaults to None.
        thresh (float): threshold for graph computation.
            Defaults to sqrt(3)+eps
        lam (float): distance function parameter
        res (list): residual information per block
        res_method (str): residual calculation method

    Returns:
        np.array: GFT matrix
        np.array: GFT frequency vector
        list: residual information per block
    """

    # Parameters
    N = V.shape[0]
    if Q is None:  # TODO: check why Q=np.ones((N,1))
        Q = np.ones((N, 1))

    # Construct graph
    if N != 1:
        D, DV, _ = graph.construct_graph_distance_encoder(V, A, lam=lam)
        W, _ = graph.construct_graph(V=V, thresh=thresh, D=D)

        # Residuals encoder
        if lam > 0:
            W_enc = W
            W_dec, _ = graph.construct_graph(V=V, thresh=thresh, D=DV)
            res = encode_residual(
                W_enc=W_enc, W_dec=W_dec, res_method=res_method,
                dist_thresh=dist_thresh)

            # Infos to be used at decoder (and decoder simulation in encoder)
            W = W_dec

        # Residuals decoder
        # Decoder also called from encoder
        if len(res) > 0 and not np.all(res == np.array([0])):
            res_mask = decode_residual(
                res=res, N=N, res_method=res_method)
            W[res_mask] = 0  # setting signalled entries to zero

            # if np.any(np.sum(W, axis=0) == 0):
            #     dbg=0

    else:  # Only one point in block
        W = np.array([1.0])
        if lam > 0:
            res = np.array([0])

    # GFT
    GFT, Gfreq = compute_gft(W, Q)  # GFT matrix

    return GFT, Gfreq, res


def transform_block_gft(
        V, A, Q, idx_start, idx_stop, ret_GFT=True, thresh=np.sqrt(3)+0.00001,
        lam=0, res_method='idx', res_thresh=0.7, dist_thresh=2):
    """ Graph Fourier transform per block

    Args:
        V (np.array): vertices
        A (np.array): attributes
        Q (np.array): node weights
        idx_start (np.array): block start indices
        idx_stop (np.array): block stop indices
        ret_GFT (bool, optional): Return GFT parameters per block.
            Defaults to True.
        thresh (float): threshold for graph computation
        lam (float): distance function parameter

    Returns:
        np.array: GFT spectra per block
        list: residual data per block
        np.array: GFT matrices per block (only if ret_GFT=True)
        np.array: GFT frequencies per block (only if ret_GFT=True)
    """

    # Iterate over all blocks
    Ahat = np.zeros_like(A, dtype='float64')  # GFT spectra
    res = []  # residual data
    if ret_GFT:
        GFT_blocks = []
        Gfreq_blocks = []

    A_std = np.std(A, axis=0)[0]

    for it in tqdm(range(len(idx_start))):
        # Get block data
        Vb = V[idx_start[it]:idx_stop[it], :]
        Ab = A[idx_start[it]:idx_stop[it], :]
        Qb = Q[idx_start[it]:idx_stop[it]]

        #
        A_std_b = np.std(Ab, axis=0)[0]
        if A_std_b > res_thresh*A_std:
            lamb = lam
        else:
            lamb = 0

        # lamb = lam

        # Construct graph and calculate GFT
        GFT_b, Gfreq_b, res_b = compute_graph_and_gft(
            V=Vb, A=Ab, Q=Qb, thresh=thresh, lam=lamb, res=[],
            res_method=res_method, dist_thresh=dist_thresh)

        # Actual transform
        Ahat[idx_start[it]:idx_stop[it], :] = transform_gft(
            X=Ab, GFT=GFT_b)

        # Append
        if ret_GFT:
            GFT_blocks.append(GFT_b)
            Gfreq_blocks.append(Gfreq_b)

        res.append(res_b)

    # Return values
    if ret_GFT:
        return Ahat, res, GFT_blocks, Gfreq_blocks
    else:
        return Ahat, res


def itransform_block_gft(
        V, Ahat, Q, idx_start, idx_stop, GFT_blocks=None,
        thresh=np.sqrt(3)+0.00001, res=[], res_method='idx'):
    """ Inverse graph Fourier transform per block

    Args:
        V (np.array): vertices
        Ahat (np.array): attributes
        Q (np.array): node weights
        idx_start (np.array): block start indices
        idx_stop (np.array): block stop indices
        GFT_blocks (list, optional): GFT matrices per block. Defaults to None.
        res (list, optional): Residual information per block
        th (float): threshold for graph computation

    Returns:
        np.array: graph signal
    """

    if len(res) == 0:
        res = [[]]*V.shape[0]

    # Iterate over all blocks
    A = np.zeros_like(Ahat, dtype='float64')
    for it in tqdm(range(len(idx_start))):
        # Get block data
        Vb = V[idx_start[it]:idx_stop[it], :]
        Ahat_b = Ahat[idx_start[it]:idx_stop[it], :]
        Qb = Q[idx_start[it]:idx_stop[it]]
        res_b = res[it]

        # Construct graph and calculate GFT
        if GFT_blocks is None:
            GFT_b, _, _ = compute_graph_and_gft(
                V=Vb, A=None, Q=Qb, thresh=thresh, lam=0,
                res=res_b, res_method=res_method)
        else:
            GFT_b = GFT_blocks[it]

        # Inverse transform
        A[idx_start[it]:idx_stop[it], :] = itransform_gft(
            Xhat=Ahat_b, GFT=GFT_b)

    return A


def create_sort_masks_subgraphs(Gfreq_blocks, idx_start, idx_stop, N_block,
                                sort_method='dc_subgraphs'):
    """ Create masks to sort GFT coefficients

    Args:
        Gfreq_blocks (np.array): GFT frequencies per block
        idx_start (np.array): block start indices
        idx_stop (np.array): block stop indices
        N_block (np.array): number of points per block
        sort_method (str, optional): sort method. Defaults to 'dc_subgraphs'.

    Returns:
        np.array: lowpass coeff mask
        np.array: highpass coeff mask
        np.array: number of subgraphs per block
    """

    N = np.sum(N_block)  # number of total points
    mask_lo = np.zeros((N), dtype=bool)  # mask indicating DC coefficients
    num_subgraphs_blocks = np.ones_like(N_block)  # number of subgraphs

    if sort_method == 'dc_subgraphs':
        # Sort DC coefficients per blocks and subgraphs to the front
        for block_id in range(len(N_block)):

            # Get number of subgraphs and mask indicating the position of DC
            num_subgraphs_b, mask_lo_b = graph.num_subgraphs_from_gfreq(
                Gfreq_blocks[block_id], return_mask=True)

            # Number of subgraphs
            num_subgraphs_blocks[block_id] = num_subgraphs_b

            # DC-values
            mask_lo[idx_start[block_id]:idx_stop[block_id]] = mask_lo_b

    elif sort_method == 'dc':
        # Sort only DC per blocks to the front
        mask_lo[idx_start] = True

    else:
        # No sorting
        mask_lo = np.ones((N), dtype=bool)

    # High-pass values
    mask_hi = np.logical_not(mask_lo)

    return mask_lo, mask_hi, num_subgraphs_blocks


def sort_block_gft_coeffs(Ahat, Gfreq_blocks, idx_start, idx_stop, N_block,
                          sort_method='dc_subgraphs'):
    """ Sort GFT coefficients

    Args:
        Ahat (np.array): GFT coefficients
        Gfreq_blocks (np.array): GFT frequencies
        idx_start (np.array): block start indices
        idx_stop (np.array): block_stop_indices
        N_block (np.array): number of points per block
        sort_method (str, optional): sort method. Defaults to 'dc_subgraphs'.

    Returns:
        np.array: sorted GFT coefficients
        np.array: lowpass coeff mask
        np.array: highpass coeff mask
        np.array: number of subgraphs per block
    """
    mask_lo, mask_hi, num_subgraphs_blocks = create_sort_masks_subgraphs(
        Gfreq_blocks=Gfreq_blocks,
        idx_start=idx_start, idx_stop=idx_stop, N_block=N_block,
        sort_method=sort_method)

    Ahat_lo = Ahat[mask_lo, :]  # DC values
    Ahat_hi = Ahat[mask_hi, :]  # "high" pass values

    # Concatenate
    Ahat_sort = np.concatenate((Ahat_lo, Ahat_hi))

    return Ahat_sort, mask_lo, mask_hi, num_subgraphs_blocks


def reverse_sort_block_gft_coeffs(Ahat_sort, mask_lo, mask_hi):
    """Reverse sorting

    Args:
        Ahat_sort (np.array): sorted array
        mask_lo (np.array): lowpass mask
        mask_hi (np.array): highpass mask

    Returns:
        np.array: array with reversed sorting
    """

    N_lo = np.sum(mask_lo)  # number of lowpass coefs
    Ahat_lo_rsort = Ahat_sort[0:N_lo, :]  # All lowpass coefs
    Ahat_hi_rsort = Ahat_sort[N_lo:, :]  # All highpass coefs

    # Reconstruct
    Ahat_rsort = np.zeros((Ahat_sort.shape[0], 3))
    Ahat_rsort[mask_lo, :] = Ahat_lo_rsort
    Ahat_rsort[mask_hi, :] = Ahat_hi_rsort

    return Ahat_rsort


def encode_residual(W_enc, W_dec, res_method='new', dist_thresh=2):

    # Differences
    res_mat = np.logical_xor(W_dec, W_enc)

    # Transmit only "significant" changes
    # At least for smaller lambda, more entries in W_enc are set to
    # zero compared to W_dec. This is because W~1/D and D is larger
    # for lambda>0
    res_mat[W_dec < dist_thresh] = 0

    # One half triangle is enough info, thanks to symmetry
    res_mat = np.triu(res_mat)

    if res_method == 'mat':  # use binary matrix directly
        res = res_mat
        if np.sum(res) == 0:  # handle all-zero block
            res = np.array([])

    elif res_method == 'idx':  # use only indices
        # Find indices of True entries
        res_idx = np.argwhere(res_mat.flatten()).flatten()
        if len(res_idx) == 0:
            res_idx = np.array([0])
        # elif N < 200:
        #    res_idx = np.array([0])
        # elif len(res_idx) > 1:
        #    res_idx = np.array([res_idx[0]])
        # elif len(res_idx) > 5:
        #    res_idx = res_idx[0:5]
        res = res_idx

    elif res_method == 'new':
        # Find indices of True entries
        res_idx = np.argwhere(res_mat)
        if len(res_idx) == 0:
            to_encode = np.array([0])
        else:
            delta_res_idx = np.diff(res_idx, axis=0)-1
            to_encode = np.concatenate((
                res_idx[0, :][np.newaxis, :]-1, delta_res_idx)
            ).flatten()

        res = to_encode

    return res


def decode_residual(res, N, res_method):
    if res_method == 'mat':
        res_mat = res

    elif res_method == 'idx':
        res_idx = res
        res_mat = np.zeros((N, N), dtype=bool)
        res_mat.ravel()[res_idx] = True

    elif res_method == 'new':
        to_encode = res
        to_encode = to_encode.reshape(-1, 2)
        res_idx = np.cumsum(to_encode+1, axis=0)

        res_mat = np.zeros((N, N), dtype=bool)
        res_mat[res_idx[:, 0], res_idx[:, 1]] = True

    res_mat = res_mat + res_mat.T  # revert triu operation

    return res_mat
