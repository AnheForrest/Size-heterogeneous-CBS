"""
全局时空预约表模块
记录每个时空点被哪些智能体占用，用于冲突检测和记录路径规划。
"""

from typing import Dict, Tuple, Set, List, Optional
from collections import defaultdict
from sh_agent import AgentInstance, AgentClass

class ReservationTable:
    """
    全局时空预约表。
    数据结构：table[(x, y, t)] = {'count': int, 'agents': set, 'is_bridge': bool}
    """
    def __init__(self, bridge_cells: Set[Tuple[int, int]] = None):
        """
        初始化预约表。
        :param bridge_cells: 桥栅格坐标集合，用于标记时空点是否为桥。
        """
        self.table: Dict[Tuple[int, int, int], Dict] = {}
        self.bridge_cells = bridge_cells if bridge_cells is not None else set()

    def _get_occupied_cells(self, pos: Tuple[int, int], agent_class: AgentClass) -> Set[Tuple[int, int]]:
        """根据左上角位置和智能体类别，返回占据的所有栅格坐标集合。"""
        x, y = pos
        w, h = agent_class.width, agent_class.height
        cells = set()
        for dx in range(w):
            for dy in range(h):
                cells.add((x + dx, y + dy))
        return cells

    def add(self, agent_id: int, path: List[Tuple[int, int]], agent_class: AgentClass) -> None:
        """
        将智能体的整条路径写入预约表。
        每个时间步的每个占用栅格增加计数，并记录智能体ID。
        :param agent_id: 智能体唯一标识
        :param path: 路径列表，每个元素为左上角坐标
        :param agent_class: 智能体类别，用于计算占据栅格
        """
        for t, pos in enumerate(path):
            cells = self._get_occupied_cells(pos, agent_class)
            for cx, cy in cells:
                key = (cx, cy, t)
                if key not in self.table:
                    is_bridge = (cx, cy) in self.bridge_cells
                    self.table[key] = {'count': 0, 'agents': set(), 'is_bridge': is_bridge}
                self.table[key]['count'] += 1
                self.table[key]['agents'].add(agent_id)

    def remove(self, agent_id: int, path: List[Tuple[int, int]], agent_class: AgentClass) -> None:
        """
        从预约表中删除智能体的旧路径记录。
        减少对应时空点的计数，并从agents集合中移除智能体ID。
        :param agent_id: 智能体唯一标识
        :param path: 路径列表
        :param agent_class: 智能体类别
        """
        for t, pos in enumerate(path):
            cells = self._get_occupied_cells(pos, agent_class)
            for cx, cy in cells:
                key = (cx, cy, t)
                if key not in self.table:
                    continue  #？？？理论上不应发生
                self.table[key]['count'] -= 1
                self.table[key]['agents'].discard(agent_id)
                if self.table[key]['count'] <= 0:
                    # 删除空条目
                    del self.table[key]

    def query(self, x: int, y: int, t: int) -> Dict:
        """
        查询指定时空点的预约信息。
        :return: 包含 count, agents, is_bridge 的字典，若不存在返回默认信息
        """
        key = (x, y, t)
        if key in self.table:
            return self.table[key].copy()
        else:
            # 即使该点未被预约，也可能需要知道是否为桥
            is_bridge = (x, y) in self.bridge_cells
            return {'count': 0, 'agents': set(), 'is_bridge': is_bridge}

    def get_conflicts(self) -> List[Tuple[int, int, int, Set[int]]]:
        """
        扫描表，返回顶点冲突列表。
        顶点冲突：同一时空点被多个智能体占用（count > 1）。
        :return: 列表，每个元素为 (x, y, t, agents_set)
        """
        conflicts = []
        for (x, y, t), info in self.table.items():
            if info['count'] > 1:
                conflicts.append((x, y, t, info['agents'].copy()))
        return conflicts

    def get_edge_conflicts(self, agents: List[AgentInstance]) -> List[Tuple[int, Tuple[int, int], Tuple[int, int], int, int]]:
        """
        检测边冲突（交换冲突）。
        边冲突定义：两个智能体在相邻时间步交换了位置区域。
        即存在时间 t，智能体 a 在 t 占据的区域与智能体 b 在 t+1 占据的区域有交集，
        且智能体 a 在 t+1 占据的区域与智能体 b 在 t 占据的区域有交集。

        :param agents: 智能体实例列表，每个实例需有 id, path, agent_class
        :return: 边冲突列表，每个元素为 (t, pos_a_t, pos_a_t1, agent_id_a, agent_id_b)
                 其中 pos_a_t 为智能体 a 在 t 时刻的左上角坐标，pos_a_t1 为 t+1 时刻的坐标。
                 冲突是对称的，只记录一对。
        """
        #为每个智能体预先计算每个时间步的占据栅格集合，避免重复计算
        agent_occupancy = {}  # agent_id -> {t: set_of_cells}
        for agent in agents:
            if agent.path is None:
                continue
            occ = {}
            for t, pos in enumerate(agent.path):
                occ[t] = self._get_occupied_cells(pos, agent.agent_class)
            agent_occupancy[agent.global_id] = occ  # 使用 global_id

        edge_conflicts = []
        #遍历所有智能体对
        agent_ids = list(agent_occupancy.keys())
        for i in range(len(agent_ids)):
            aid = agent_ids[i]
            a = next((ag for ag in agents if ag.global_id == aid), None)
            if a is None:
                continue
            occ_a = agent_occupancy[aid]
            path_a = a.path
            for j in range(i+1, len(agent_ids)):
                bid = agent_ids[j]
                b = next((ag for ag in agents if ag.global_id == bid), None)
                if b is None:
                    continue
                occ_b = agent_occupancy[bid]
                path_b = b.path
                if path_a is None or path_b is None:
                    continue
                #遍历可能发生交换的时间步
                max_t = min(len(path_a), len(path_b)) - 1
                for t in range(max_t):  # t 从 0 到 max_t-1
                    if t+1 in occ_a and t in occ_b and t+1 in occ_b:
                        cells_a_t = occ_a[t]
                        cells_a_t1 = occ_a[t+1]
                        cells_b_t = occ_b[t]
                        cells_b_t1 = occ_b[t+1]
                        # 检查交换条件：a_t ∩ b_t1 非空 且 a_t1 ∩ b_t 非空
                        if (cells_a_t & cells_b_t1) and (cells_a_t1 & cells_b_t):
                            # 记录冲突，可用 a 的 t 和 t+1 位置代表
                            edge_conflicts.append((
                                t,
                                path_a[t], path_a[t+1],
                                aid, bid
                            ))
        return edge_conflicts

    def clear(self):
        """清空预约表"""
        self.table.clear()

    def __repr__(self):
        return f"ReservationTable(size={len(self.table)})"