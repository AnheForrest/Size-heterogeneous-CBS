"""
拥堵系数模块
计算每个栅格被多少类智能体可通行，用于引导路径规划。
"""

from typing import Dict, List

from gridmap import GridMap
from passable_graph import PassableGraph


def compute_cr(grid_map: GridMap,
               passable_graphs: Dict[int, PassableGraph]) -> List[List[int]]:
    """
    计算每个栅格的拥堵系数，即该栅格被多少类智能体可通行。

    :param grid_map: 基底地图，提供 cols 和 rows
    :param passable_graphs: 字典，键为类别编号，值为该类智能体的 PassableGraph 对象
                             要求每个 PassableGraph 对象具有 width 和 height 属性（尺寸）
    :return: 二维列表 cr[x][y]，表示坐标 (x, y) 的拥堵系数
    """
    cols, rows = grid_map.cols, grid_map.rows
    #初始化 cr
    cr = [[0 for _ in range(rows)] for _ in range(cols)]

    for cat, pg in passable_graphs.items():
        #确保 pg 有 width 和 height 属性
        if not hasattr(pg, 'width') or not hasattr(pg, 'height'):
            raise AttributeError(f"PassableGraph for category {cat} missing width/height attributes.")
        w, h = pg.width, pg.height

        #遍历该类所有合法左上角位置
        for (x, y) in pg.V:
            #该位置覆盖的所有栅格
            for dx in range(w):
                for dy in range(h):
                    cx = x + dx
                    cy = y + dy
                    # 边界检查
                    if 0 <= cx < cols and 0 <= cy < rows:
                        cr[cx][cy] += 1
                    else:
                        #若越界则忽略
                        pass

    return cr