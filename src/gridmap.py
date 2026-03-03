"""
栅格地图模块
功能：
1.输入长宽创建栅格地图 ->__init__
2.输入障碍物比例，在栅格图中随机生成障碍 -> create_random_obstacles
3.检测位置是否可通行
"""

from typing import List ,Tuple
import random

class GridMap:

    def __init__(self , cols : int , rows : int ):
        """
        创建空地图
        Args:
            cols: 列数（水平方向）
            rows: 行数（垂直方向）
        """

        # 参数校验
        if cols <= 0 or rows <= 0:
            raise ValueError("列数和行数必须为正整数")
        
        self.cols = cols
        self.rows = rows

        #地图初始化
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]

        print(f"地图创建完成：{rows}行 * {cols}列")

    def create_random_obstacles(self , ratio : float) -> List[Tuple[int , int]]:
        """
        按比例创建随机障碍物
        Args:
            ratio: 障碍物占比[0%，100%）
        Returns:
            障碍物坐标列表
        """
        #参数校验
        if ratio<0 or ratio >=1:
            raise ValueError("障碍物占比应在[0%，100%）区间内")
        
        #计算总障碍数
        total_cells = self.cols * self.rows
        n_obstacles = int(ratio * total_cells)
        if total_cells <= n_obstacles:
            raise ValueError("障碍物占比过高，无可通行栅格")

        #随机生成障碍物坐标
        all_coords = [(x , y) for x in range(self.cols) for y in range(self.rows)]
        obstacle_coords = random.sample(all_coords , n_obstacles)

        #标记障碍物坐标的值为1
        for x , y in obstacle_coords :
            self.grid[y][x] = 1

        print(f"已生成随机障碍{n_obstacles}个，占比{ratio*100:.1f}%")

        return obstacle_coords
    
    def is_passable(self, x: int, y: int) -> bool:
        """返回 (x,y) 位置是否可通行（0表示可通行，1表示障碍）"""
        if 0 <= x < self.cols and 0 <= y < self.rows:
            return self.grid[y][x] == 0
        return False