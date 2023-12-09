import scipy.io as sio


def read_file(file_dir: str):
    mat = sio.loadmat(file_dir)

    data = dict()
    for key in mat.keys():
        if not key.startswith('__') and not key.endswith('__'):
            data[key] = mat[key]

    return data
