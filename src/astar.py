"""
A* 低层规划模块
为单个智能体规划路径，考虑拥堵系数、桥栅格、全局预约表软约束以及CBS硬约束。
"""

import heapq
from typing import List, Tuple, Optional, Dict

from sh_agent import AgentInstance
from passable_graph import PassableGraph
from reservation_table import ReservationTable


def astar(agent_instance: AgentInstance,
          passable_graph: PassableGraph,
          reservation_table: ReservationTable,
          cr: List[List[int]],
          weight_cr: float = 0.1,
          weight_bridge: float = 10.0,
          weight_res: float = 1.0,
          constraints: List[Tuple[int, Tuple[int, int]]] = None) -> Optional[List[Tuple[int, int]]]:
    """
    使用 A* 算法为单个智能体规划路径，支持 CBS 硬约束（禁止在特定时刻占据特定位置）。

    :param agent_instance: 智能体实例（包含起点、终点、尺寸）
    :param passable_graph: 该类智能体的可通行子图
    :param reservation_table: 全局时空预约表
    :param cr: 拥堵系数矩阵，二维列表 cr[x][y]（x 为列索引，y 为行索引）
    :param weight_cr: 拥堵系数权重
    :param weight_bridge: 桥栅格惩罚权重（每个桥栅格累加）
    :param weight_res: 预约表占用计数权重
    :param constraints: 硬约束列表，每个元素为 (t, pos)，表示禁止在时间 t 占据位置 pos
    :return: 路径列表（每个元素为左上角坐标），若无解则返回 None
    """
    if constraints is None:
        constraints = []

    start = agent_instance.start
    goal = agent_instance.goal
    agent_class = agent_instance.agent_class
    width, height = agent_class.width, agent_class.height

    # 快速连通性检查
    if not passable_graph.is_connected(start, goal):
        return None

    # 构建硬约束字典，方便快速查找
    constraint_dict = {}
    for t, pos in constraints:
        if t not in constraint_dict:
            constraint_dict[t] = set()
        constraint_dict[t].add(pos)

    # 辅助函数：获取智能体在某个左上角位置占据的所有栅格
    def get_occupied_cells(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        cells = []
        for dx in range(width):
            for dy in range(height):
                cells.append((x + dx, y + dy))
        return cells

    # 计算在特定时刻占据特定位置的单步惩罚（包括拥堵、桥、预约）
    def step_penalty(pos: Tuple[int, int], t: int) -> float:
        # 检查硬约束
        if t in constraint_dict and pos in constraint_dict[t]:
            return float('inf')  # 不可行状态
        cells = get_occupied_cells(pos)
        total = 0.0
        for cx, cy in cells:
            # 拥堵系数
            total += weight_cr * cr[cx][cy]
            # 桥栅格
            if (cx, cy) in passable_graph.bridges:
                total += weight_bridge
            # 预约表占用
            info = reservation_table.query(cx, cy, t)
            total += weight_res * info['count']
        return total

    # 启发函数（曼哈顿距离）
    def heuristic(pos: Tuple[int, int]) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # 优先队列元素：(f, g, t, pos)
    open_set = []
    # 初始状态
    t0 = 0
    g0 = step_penalty(start, t0)
    if g0 == float('inf'):  # 起点被硬约束禁止
        return None
    f0 = g0 + heuristic(start)
    heapq.heappush(open_set, (f0, g0, t0, start))

    g_score: Dict[Tuple[Tuple[int, int], int], float] = {(start, t0): g0}
    came_from: Dict[Tuple[Tuple[int, int], int], Tuple[Tuple[int, int], int]] = {}

    while open_set:
        f, g, t, pos = heapq.heappop(open_set)

        if pos == goal:
            # 重建路径
            path = []
            state = (pos, t)
            while state in came_from:
                path.append(state[0])
                state = came_from[state]
            path.append(start)
            path.reverse()
            return path

        if g > g_score.get((pos, t), float('inf')):
            continue

        # 扩展邻居（四向移动）
        for nbr in passable_graph.get_neighbors(pos):
            nt = t + 1
            penalty = step_penalty(nbr, nt)
            if penalty == float('inf'):
                continue
            ng = g + 1 + penalty
            state = (nbr, nt)
            if ng < g_score.get(state, float('inf')):
                g_score[state] = ng
                f_val = ng + heuristic(nbr)
                heapq.heappush(open_set, (f_val, ng, nt, nbr))
                came_from[state] = (pos, t)

        # 扩展等待动作
        nt = t + 1
        penalty = step_penalty(pos, nt)
        if penalty != float('inf'):
            ng = g + 1 + penalty
            state = (pos, nt)
            if ng < g_score.get(state, float('inf')):
                g_score[state] = ng
                f_val = ng + heuristic(pos)
                heapq.heappush(open_set, (f_val, ng, nt, pos))
                came_from[state] = (pos, t)

    return None