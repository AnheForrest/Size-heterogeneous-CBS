"""
可通行子图模块
为每类智能体生成专属的可通行子图，包含顶点集、边集、桥栅格检测等功能。
"""

from typing import List, Tuple, Set, Dict, Optional
from collections import deque
from gridmap import GridMap
from sh_agent import AgentClass

class PassableGraph:
    """
    每类智能体的可通行子图。
    """
    def __init__(self, category: int):
        """
        初始化一个空的子图。

        :param category: 智能体类别编号
        """
        self.category = category
        self.V: List[Tuple[int, int]] = []               #顶点列表（合法左上角坐标）
        self.E: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}  #可达矩阵邻接表
        self.bridges: Set[Tuple[int, int]] = set()       #桥栅格集合（物理栅格坐标）
        self._pos_to_index: Dict[Tuple[int, int], int] = {}  # 位置到索引的映射，用于桥检测

    def build_from(self, grid_map: GridMap, agent_class: AgentClass) -> 'PassableGraph':
        """
        根据基底地图和智能体类别生成可通行子图。

        :param grid_map: 基底地图
        :param agent_class: 智能体类别
        :return: self
        """
        cols, rows = grid_map.cols, grid_map.rows
        w, h = agent_class.width, agent_class.height

        # 1. 枚举所有可能的左上角坐标，检查合法性
        V = []
        for x in range(cols - w + 1):
            for y in range(rows - h + 1):
                if self._is_position_valid(grid_map, (x, y), agent_class):
                    V.append((x, y))

        self.V = V
        self._pos_to_index = {pos: idx for idx, pos in enumerate(V)}

        # 2. 构建邻接表（只考虑四向移动）
        neighbors_map = {}
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上、下、左、右
        for pos in V:
            x, y = pos
            nbrs = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                nbr_pos = (nx, ny)
                if nbr_pos in self._pos_to_index:
                    nbrs.append(nbr_pos)
            neighbors_map[pos] = nbrs
        self.E = neighbors_map

        return self

    def _is_position_valid(self, grid_map: GridMap, pos: Tuple[int, int],
                           agent_class: AgentClass) -> bool:
        """
        判断给定左上角位置是否合法（智能体矩形不碰障碍物且不越界）。

        :param grid_map: 基底地图
        :param pos: 左上角坐标 (x, y)
        :param agent_class: 智能体类别（提供尺寸）
        :return: 合法返回 True
        """
        x, y = pos
        w, h = agent_class.width, agent_class.height
        # 边界检查
        if x < 0 or y < 0 or x + w > grid_map.cols or y + h > grid_map.rows:
            return False
        for dx in range(w):
            for dy in range(h):
                if grid_map.grid[y + dy][x + dx] == 1:  # 障碍物
                    return False
        return True

    def find_bridges(self, agent_class: AgentClass) -> Set[Tuple[int, int]]:
        """
        使用 Tarjan 算法找出子图中的桥（割边），并记录桥栅格。
        桥栅格定义为：桥边所涉及的两个顶点（位置）覆盖的所有物理栅格的并集。

        :param agent_class: 智能体类别（用于获取覆盖栅格）
        :return: 桥栅格集合（物理栅格坐标）
        """
        if not self.V:
            return set()

        n = len(self.V)
        idx = self._pos_to_index
        adj = {pos: [nbr for nbr in self.E[pos]] for pos in self.V}  # 邻接表

        # Tarjan 算法找桥
        ids = [-1] * n
        low = [0] * n
        visited = [False] * n
        bridges_edges = []  # 存储桥边 (u, v)

        def dfs(at, parent, depth):
            visited[at] = True
            ids[at] = low[at] = depth
            pos_u = self.V[at]
            for v_pos in adj[pos_u]:
                v = idx[v_pos]
                if v == parent:
                    continue
                if not visited[v]:
                    dfs(v, at, depth + 1)
                    low[at] = min(low[at], low[v])
                    if ids[at] < low[v]:
                        bridges_edges.append((pos_u, v_pos))
                else:
                    low[at] = min(low[at], ids[v])

        for i in range(n):
            if not visited[i]:
                dfs(i, -1, 0)

        #将桥边涉及的顶点覆盖的所有栅格加入桥栅格集合
        bridge_cells = set()
        for u_pos, v_pos in bridges_edges:
            for pos in [u_pos, v_pos]:
                cells = agent_class.get_occupied_cells(pos)
                bridge_cells.update(cells)

        self.bridges = bridge_cells
        return bridge_cells

    def is_connected(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """
        判断起点和终点是否在同一连通分量中。

        :param start: 起点左上角坐标
        :param goal: 终点左上角坐标
        :return: 连通返回 True，否则 False
        """
        if start not in self._pos_to_index or goal not in self._pos_to_index:
            return False
        # BFS
        visited = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            pos = queue.popleft()
            if pos == goal:
                return True
            for nbr in self.E.get(pos, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return False

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        返回从该位置出发通过一次移动可到达的邻居位置列表。

        :param pos: 左上角坐标
        :return: 邻居位置列表（可能为空）
        """
        return self.E.get(pos, [])

    def __repr__(self) -> str:
        return (f"PassableGraph(cat={self.category}, |V|={len(self.V)}, "
                f"|E|={sum(len(v) for v in self.E.values())//2}, |bridges|={len(self.bridges)})")