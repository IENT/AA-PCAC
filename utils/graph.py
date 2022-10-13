# Parts of this source file are adapted from https://github.com/STAC-USC/RA-GFT

import numpy as np
import scipy.spatial

import utils.color as color

def block_indices(V, bsize):
    """Generate block indices. Assumes Morton-ordering.

    Args:
        V (np.array): vertices, xyz coordinates of each points
        bsize (int): block size

    Returns:
        _type_: _description_
    """

    if not np.log2(bsize).is_integer():
        raise ValueError

    # Quantize coordinates
    V_coarse = np.floor(V/bsize)*bsize

    # Find block limits
    variation = np.sum(np.abs(V_coarse[1:, :] - V_coarse[0:-1, :]), axis=1)
    variation = np.concatenate(([1], variation))

    idx_start = np.argwhere(variation)
    idx_stop = np.row_stack((idx_start[1:], [V.shape[0]]))

    idx_start = idx_start.flatten()
    idx_stop = idx_stop.flatten()

    # Number of points per block
    num_per_block = idx_stop - idx_start

    return idx_start, idx_stop, num_per_block


def construct_graph(V, thresh=np.sqrt(3)+0.00001, D=None):
    """Construct graph from vertices

    Args:
        V (np.array): vertices of shape (n, 3)
        th (double, optional): threshold to construct the graph.
            Defaults to np.sqrt(3)+0.00001. If given, distances
            above threshold are set to zero
        D (np.array): Pre-calculated distances

    Returns:
        np.array: weight matrix
        np.array: edges
    """

    if D is None:
        D, _, _ = construct_graph_distance_encoder(V=V, A=None, lam=0)

    iD = 1/D  # element-wise
    iD[D == 0] = 0  # 1/0 elements on diagonal

    if thresh is not None:
        iD[D > thresh] = 0  # discard everything above threshold

    # Adjacency matrix
    W = iD.T + iD  # TODO: why add iD.T? Perhaps ensure symmetry?

    # Edges
    idx = np.argwhere(iD != 0)
    I, J = np.unravel_index(idx, D.shape)
    edge = np.concatenate((I, J), axis=0)

    return W, edge


def construct_graph_distance_encoder(V, A, lam=0):
    """Calculate pairwise Euclidean distance

    Args:
        V (np.array): vertices
        A (np.array): attributes, only available at encoder
        lam (int, optional): distance steering parameter. Defaults to 0.

    Returns:
        np.array: distance matrices
    """
    # Pairwise distances
    # V should be m by n array of m original observations in an
    # n-dimensional space.
    # If scipy is not installed:
    # N = V.shape[0]
    # squared_norms = np.sum(V**2, axis=1)[:, np.newaxis]
    # D2 = np.sqrt(
    #     np.tile(squared_norms, (1, N)) +
    #     np.tile(squared_norms.T, (N, 1)) - 2*(V@V.T)
    # )

    # Pairwise distances
    # Vertices
    DV = scipy.spatial.distance.squareform(
         scipy.spatial.distance.pdist(V, 'Euclidean')
    )

    # Y color component
    if lam > 0:
        Ycomponent = A[:, 0][:, np.newaxis]  # n x shape
        DA = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(Ycomponent, 'Euclidean')
        )

        D = (1-lam)*DV + lam*DA
    else:
        DA = 0
        D = DV

    return D, DV, DA


def w2l(W):
    """weight matrix to Laplacian matrix

    Args:
        W (np.array): weight matrix

    Returns:
        np.array: Laplacian
    """
    # by: KS Lu 20170712

    if np.any(W < 0):
        raise ValueError('W is not a valid weight matrix')

    # Degree matrix
    D = np.diag(np.sum(W, axis=0))

    # Laplacian
    L = D - W + np.diag(np.diag(W))

    return L


def num_subgraphs_from_gfreq(Gfreq, thresh=1e-10, return_mask=False):
    """Compute number of subgraphs within subgraph, assuming GFT matrix and
        GFT frequencies were already calculated

    Args:
        Gfreq (np.array): GFT frequencies
        thresh (float, optional): threshold. Defaults to 1e-10.

    Returns:
        int: number of subgraphs
    """

    # If multiple frequencies are very close to 0, we have multiple subgraphs
    mask = np.abs(Gfreq) < thresh
    num_subgraphs = np.sum(mask)

    if not return_mask:
        return num_subgraphs
    else:
        return num_subgraphs, mask


def compute_subgraphs_ids(W):
    """Extract indices of points belonging to subgraphs/components

    Args:
        W (np.array): graph weight matrix

    Returns:
        list: list of indices of points per subgraph/component
    """

    # Taken from
    # https://pygsp.readthedocs.io/en/latest/_modules/pygsp/graphs/graph.html#Graph.extract_components
    # A = scipy.sparse.csr_matrix(W > 0.0)

    A = W > 0  # Adjacency boolean matrix

    if A.shape[0] != A.shape[1]:
        return None

    ids = []

    visited = np.zeros(A.shape[0], dtype=bool)
    # indices = [] # Assigned but never used

    while not visited.all():
        # pick a node not visited yet
        stack = set(np.nonzero(~visited)[0][[0]])
        ids_component = []

        while len(stack):
            id = stack.pop()
            if not visited[id]:
                ids_component.append(id)
                visited[id] = True

                # Add indices of nodes not visited yet and accessible from id
                stack.update(set(
                    [idx for idx in A[id, :].nonzero()[0] if not visited[idx]]
                ))

        ids_component = sorted(ids_component)  # sort indices
        # print(('Constructing subgraph for component of '
        #                     'size {}.').format(len(comp)))
        ids.append(ids_component)

    return ids


def plot_points(V, A, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', aspect='auto')

    sc = ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=A/255.0)
    plt.show()

    return fig, ax, sc


def ypsnr_per_block(A, Aquant, idx_start, idx_stop, N_block):

    YPSNR_block = np.zeros(len(N_block))
    for block_id in range(len(N_block)):
        Ab = A[idx_start[block_id]:idx_stop[block_id], :]
        Aquantb = Aquant[idx_start[block_id]:idx_stop[block_id], :]

        YPSNR_block[block_id] = color.YPSNR(
            A=Ab, Aq=Aquantb, N=N_block[block_id])

    return YPSNR_block
