import scipy.io as sio


def read_file(file_dir: str):
    """ 从指定位置读取 .mat文件

    :param file_dir: 需要读取的文件的位置
    :return: 读取出的数据
    """
    mat = sio.loadmat(file_dir)

    data = dict()
    for key in mat.keys():
        if not key.startswith('__') and not key.endswith('__'):
            data[key] = mat[key]

    return data
