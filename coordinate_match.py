import pandas as pd
import dask.dataframe as dd
import numpy as np

from scipy.spatial import cKDTree
from pykdtree.kdtree import KDTree


def load_csv_data(path: str) -> pd.DataFrame:
    """加载csv数据
    parameters
    ----------
        path：str, as 'E:/BaiduNetdiskDownload/nyc_taxi_data.csv'
    return
    ------
        pd.DataFrame
    """
    data = dd.read_csv(path)
    return data.compute()


def generate_coords(bbox: list, interval: int) -> np.array:
    """卡迪尔坐标，从左到右、从上到下
    parameters
    ----------
        bbox: list,as [(min_x,min_y),(max_x, max_y)]
        interval: int, as 32
    return
    ------
         list,grid coords like[(x, y),...,(x, y)]
    ``
    bbox = [(0, 0),(1, 1)]
    generate_coords(bbox, int)
    ``
    """
    p1, p2 = bbox
    a = np.linspace(p1[0], p2[0], interval)
    b = np.linspace(p1[1], p2[1], interval)
    x, y = np.meshgrid(a, b)
    return np.c_[x.ravel(), y.ravel()]


def space_nearest_join(points: np.array, source='pykdtree', leafsize=8) -> np.array:
    """采用空间最近邻匹配的方式将提取的特征数据集匹配到位置点上, scipy
    parameter
    --------
        points: np.array, as [1, 1]
        source: str, ['pykdtree', ckdtree]
        tar_points: np.array, as [1, 1]
    return
    ------
        ind: np.array, as [1]
    ``
    call = space_nearest_join(points)
    ind = call(tar_points)
    ``
    """
    if source == 'pykdtree':
        kd_tree = KDTree(points, leafsize=leafsize)
    else:
        kd_tree = cKDTree(points, leafsize=leafsize)

    def _call(tar_points: np.array):
        _, ind = kd_tree.query(tar_points, k=1)
        return ind

    return _call


def main():
    # 数据
    data_path = './nyc_taxi_data.csv'
    save_path = './nyc_taxi_grid_data.csv'
    bbox = [(0, 0), (1, 1)]
    # 加载数据，构建tree
    nyc_taxi_data = load_csv_data(data_path)
    points = generate_coords(bbox, 32)
    call = space_nearest_join(points)
    # 最邻近点匹配
    up_points = nyc_taxi_data.loc[:, ['up_x', 'up_y']].values
    ind_up = call(up_points)
    off_points = nyc_taxi_data.loc[:, ['off_x', 'off_y']].values
    ind_off = call(off_points)
    # groupby
    nyc_taxi_data['ind_up'], nyc_taxi_data['ind_off'] = ind_up, ind_off
    up_passengers = nyc_taxi_data['passengers'].groupby(nyc_taxi_data['ind_up']).sum().rename('up_passengers')
    off_passengers = nyc_taxi_data['passengers'].groupby(nyc_taxi_data['ind_off']).sum().rename('off_passengers')

    # 保存
    grid_index = pd.Series(np.arange(0, 1024), index=np.arange(0, 1024), name='grid_index')
    nyc_taxi_grid_data = pd.concat([grid_index, up_passengers, off_passengers], axis=1)
    nyc_taxi_grid_data['grid_index'] = nyc_taxi_grid_data.loc[:, 'grid_index'] + 1  # 由于返回结果是1开头, 所以加1
    nyc_taxi_grid_data.fillna(0, inplace=True)  # 缺失值填充
    nyc_taxi_grid_data.to_csv(save_path, index=False, )


if __name__ == '__main__':
    from time import time

    t1 = time()
    main()

    print(time() - t1)
