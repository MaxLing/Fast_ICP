import numpy as np
from sklearn.neighbors import NearestNeighbors

def pc_normals(p, k=4):
    '''
    :param p: point cloud m*3
    :param k: Knn
    :return: normals with ambiguous orientaion (PCA solution problem)
    '''
    knn = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(p)
    _, index = knn.kneighbors(p)

    m = p.shape[0]
    normals = np.zeros((m,3))
    for i in range(m):
        nn = p[index[i,1:]] # exclude self in nn
        c = np.cov(nn.T)
        w,v = np.linalg.eig(c)
        normals[i] = v[:,np.argmin(w)]
    return normals

def normal_sampling(normals, nums):
    '''
    :param normals: point cloud normals m*3
    :param nums: num of samples n
    :return: index of samples
    '''

    # convert to angular space, [-pi, pi]
    azimuth = np.arctan2(normals[:,1], normals[:,0])
    altitude = np.arctan2(normals[:,2], np.sqrt(normals[:,0]**2+normals[:,1]**2))

    # compute bins in 2d and combine
    bins = 500
    index1 = np.digitize(azimuth, np.linspace(-np.pi, np.pi, bins), right=True)
    index2 = np.digitize(altitude, np.linspace(-np.pi, np.pi, bins), right=True)
    index = index1*bins+index2

    # get unique and then uniform sampling
    unique_index, origin_index = np.unique(index, return_index=True)
    sample = np.random.choice(unique_index.shape[0], size=nums, replace=False)
    sample_index = origin_index[sample]

    return sample_index


def icp(p, q, error, sample, normals = None):
    '''
    :param p: start point cloud m*3
    :param q: destination point cloud m*3
    :param normals: normals of q, call pc_normals() to obtain before this function
    :param error: 'point2point', 'point2plane'
    :param sample: 'random', 'normal_space'
    :return: current estimate of R and T, q = R*p + T
    '''
    if normals is None:
        assert error!='point2plane'

    # selection
    m = p.shape[0]
    r = 0.2
    if sample == 'random':
        index = np.random.choice(range(m), size=int(m*r), replace=False)
        p = p[index]
    elif sample == 'normal_space':
        p_nums = int(m*r)
        p_normals = pc_normals(p)
        index = normal_sampling(p_normals, p_nums)
        p = p[index]
    else: # 'all'
        pass

    # matching
    knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(q)
    dist, index = knn.kneighbors(p)
    q = q[index.reshape(-1)]


    if error == 'point2point':
        p_mean = np.mean(p,axis=0)
        q_mean = np.mean(q,axis=0)

        p_prime = p - p_mean
        q_prime = q - q_mean

        H = np.dot(p_prime.T, q_prime)
        U, D, V = np.linalg.svd(H) # note V is already transposed H = U*D*V

        R = np.dot(V.T , U.T)
        T = q_mean.reshape((-1,1)) - np.dot(R, p_mean.reshape((-1,1)))
        return R, T

    elif error == 'point2plane':
        normals = normals[index.reshape(-1)]

        b = np.sum(normals*(q-p), axis = 1)
        # A = np.block([[(normals[:, 2] * p[:, 1] - normals[:, 1] * p[:, 2]).reshape((-1,1)),
        #                (normals[:, 0] * p[:, 2] - normals[:, 2] * p[:, 0]).reshape((-1,1)),
        #                (normals[:, 1] * p[:, 0] - normals[:, 0] * p[:, 1]).reshape((-1,1)),
        #                 normals]])
        A = np.block([[np.cross(p,normals) , normals]])

        x = np.linalg.lstsq(A,b)[0] # solve least square Ax = b

        T = x[3:].reshape((-1,1))
        cx = np.cos(x[0])
        sx = np.sin(x[0])
        cy = np.cos(x[1])
        sy = np.sin(x[1])
        cz = np.cos(x[2])
        sz = np.sin(x[2])
        R = np.block([[cz*cy, -sz*cx+cz*sy*sx, sz*sx+cz*sy*cx],
                      [sz*cy, cy*cx+sz*sy*sx, -cz*sx+sz*sy*cx],
                      [-sy, cy*sx, cy*cx]])
        return R, T
