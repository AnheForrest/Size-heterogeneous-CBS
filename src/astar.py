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
          weight_bridge: float = 0.3,
          weight_res: float = 0.4,
          constraints: List[Tuple[int, Tuple[int, int]]] = None,
          wait_cost: float = 1.0 ) -> Optional[List[Tuple[int, int]]]:
    """
    使用 A* 算法为单个智能体规划路径，支持 CBS 硬约束（禁止在特定时刻占据特定位置）。
    路径代价计算移动次数，而非时间步总数。

    :param agent_instance: 智能体实例（包含起点、终点、尺寸）
    :param passable_graph: 该类智能体的可通行子图
    :param reservation_table: 全局时空预约表
    :param cr: 拥堵系数矩阵，二维列表 cr[x][y]（x 为列索引，y 为行索引）
    :param weight_cr: 拥堵系数权重
    :param weight_bridge: 桥栅格惩罚权重（每个桥栅格累加）
    :param weight_res: 预约表占用计数权重
    :param constraints: 硬约束列表，每个元素为 (t, pos)，表示禁止在时间 t 占据位置 pos
    :param wait_cost: 等待动作的步时代价（默认1.0，与移动相同；设为小于1可鼓励等待）
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

    # 辅助函数：获取智能体在某个左上角位置占据的所有栅格（这个在很多地方都用到了，直接整理到可通行图里面每次调用吧）
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

        # 分别累计三项惩罚并加权求和，限制总软约束惩罚小于1
        sum_cr = 0.0
        sum_bridge = 0.0
        sum_res = 0.0
    
        for cx, cy in cells:
            sum_cr += cr[cx][cy]
            if (cx, cy) in passable_graph.bridges:
                sum_bridge += 1
            info = reservation_table.query(cx, cy, t)
            sum_res += info['count']
    
        penalty = (weight_cr * sum_cr) + (weight_bridge * sum_bridge) + (weight_res * sum_res)
        penalty = min(penalty, 0.999)  # 确保小于移动一步的代价1
    
        return penalty

    # 启发函数（曼哈顿距离）
    def heuristic(pos: Tuple[int, int]) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # 优先队列元素：(f, g, t, pos)
    # g 表示路径代价（移动次数），t 表示时间步
    open_set = []
    # 初始状态
    t0 = 0
    penalty0 = step_penalty(start, t0)
    if penalty0 == float('inf'):  # 起点被硬约束禁止
        return None
    g0 = penalty0  # 初始g值为penalty
    f0 = g0 + heuristic(start)
    heapq.heappush(open_set, (f0, g0, t0, start))

    # 状态: (pos, t) -> g_value
    g_score: Dict[Tuple[Tuple[int, int], int], float] = {(start, t0): g0}
    # 记录父节点: (pos, t) -> ((parent_pos, parent_t), action_is_wait)
    came_from: Dict[Tuple[Tuple[int, int], int], Tuple[Tuple[Tuple[int, int], int], bool]] = {}

    while open_set:
        f, g, t, pos = heapq.heappop(open_set)

        # 到达终点后，继续等待直到预约表中不再有冲突
        if pos == goal:
            # 找到终点后，继续等待，直到到达终点的智能体不会与其他智能体冲突
            # 但我们返回的是到达终点的路径，不包括无限等待
            # 重建路径
            path = []
            state = (pos, t)
            while state in came_from:
                path.append(state[0])
                state, _ = came_from[state]  # unpack parent state and action type
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
            # 移动的代价是1（移动次数）+ penalty(惩罚）)
            ng = g + 1 + penalty  
            state = (nbr, nt)
            if ng < g_score.get(state, float('inf')):
                g_score[state] = ng
                f_val = ng + heuristic(nbr)
                heapq.heappush(open_set, (f_val, ng, nt, nbr))
                came_from[state] = ((pos, t), False)  # False代表发生了移动

        # 扩展等待动作（使用可调的等待代价）
        nt = t + 1
        penalty = step_penalty(pos, nt)
        if penalty != float('inf'):
            # 等待的代价是wait_cost + penalty
            ng = g + wait_cost + penalty
            state = (pos, nt)
            if ng < g_score.get(state, float('inf')):
                g_score[state] = ng
                f_val = ng + heuristic(pos)
                heapq.heappush(open_set, (f_val, ng, nt, pos))
                came_from[state] = ((pos, t), True)  # True代表在原地等待

    return None


"""
经典A*无软约束版
"""
def astar_classic(agent_instance, grid_map, constraints=None):
    """
    经典CBS低层A*：质点模型，1×1尺寸，仅遵守硬约束
    """
    start = agent_instance.start
    goal = agent_instance.goal
    if constraints is None:
        constraints = []
    # 构建硬约束字典
    constraint_dict = {}
    for t, pos in constraints:
        if t not in constraint_dict:
            constraint_dict[t] = set()
        constraint_dict[t].add(pos)

    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_neighbors(pos):
        x, y = pos
        candidates = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        return [p for p in candidates if grid_map.is_passable(p[0], p[1])]

    open_set = []
    t0 = 0
    if t0 in constraint_dict and start in constraint_dict[t0]:
        return None
    heapq.heappush(open_set, (heuristic(start), 0, t0, start))
    g_score = {(start, t0): 0}
    came_from = {}

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
        nt = t + 1
        # 移动
        for nbr in get_neighbors(pos):
            if nt in constraint_dict and nbr in constraint_dict[nt]:
                continue
            ng = g + 1
            state = (nbr, nt)
            if ng < g_score.get(state, float('inf')):
                g_score[state] = ng
                heapq.heappush(open_set, (ng + heuristic(nbr), ng, nt, nbr))
                came_from[state] = (pos, t)
        # 等待
        if nt not in constraint_dict or pos not in constraint_dict[nt]:
            ng = g + 1
            state = (pos, nt)
            if ng < g_score.get(state, float('inf')):
                g_score[state] = ng
                heapq.heappush(open_set, (ng + heuristic(pos), ng, nt, pos))
                came_from[state] = (pos, t)
    return None