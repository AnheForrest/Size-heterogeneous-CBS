"""
冲突检测模块
功能：
1. 从预约表中检测顶点冲突和边冲突
2. 为冲突分配严重程度
3. 按严重程度排序冲突
"""

from typing import List, Dict, Tuple, Callable, Optional
from reservation_table import ReservationTable
from sh_agent import AgentInstance


def detect_conflicts(reservation_table: ReservationTable,
                     agents: List[AgentInstance],
                     paths_override: Optional[Dict[int, List]] = None) -> List[Dict]:
    """
    从预约表中检测所有冲突，返回冲突列表。
    如果提供了 paths_override，则使用其中的路径代替 agent.path 用于边冲突检测。

    冲突格式：
        - 顶点冲突：{'type': 'vertex', 'time': t, 'pos': (x, y), 'agents': [id1, id2, ...]}
        - 边冲突：{'type': 'edge', 'time': t, 'pos': ((x1,y1),(x2,y2)), 'agents': [id_a, id_b]}

    :param reservation_table: 全局时空预约表
    :param agents: 所有智能体实例列表
    :param paths_override: 可选，覆盖智能体路径的字典，键为global_id，值为路径列表
    :return: 冲突列表
    """
    conflicts = []

    # 1. 顶点冲突（来自预约表）
    vertex_conflicts = reservation_table.get_conflicts()  # 返回 (x, y, t, agents_set)
    for (x, y, t, agents_set) in vertex_conflicts:
        conflicts.append({
            'type': 'vertex',
            'time': t,
            'pos': (x, y),
            'agents': list(agents_set)   # agents_set 中包含全局ID
        })

    # 2. 边冲突：基于路径信息计算
    # 构建每个智能体每个时间步的占用栅格集合
    agent_occupancy = {}
    path_map = {}  # aid -> path list
    for agent in agents:
        aid = agent.global_id
        if paths_override is not None and aid in paths_override:
            path = paths_override[aid]
        else:
            path = agent.path
        if path is None:
            continue
        path_map[aid] = path
        occ = {}
        for t, pos in enumerate(path):
            cells = agent.agent_class.get_occupied_cells(pos)
            occ[t] = set(cells)
        agent_occupancy[aid] = occ

    # 遍历所有智能体对
    ids = list(agent_occupancy.keys())
    for i in range(len(ids)):
        aid = ids[i]
        occ_a = agent_occupancy[aid]
        path_a = path_map[aid]
        for j in range(i+1, len(ids)):
            bid = ids[j]
            occ_b = agent_occupancy[bid]
            path_b = path_map[bid]
            # 取最大公共时间范围
            max_t = min(len(path_a), len(path_b)) - 1
            for t in range(max_t):
                # 确保 t 和 t+1 都存在
                if t not in occ_a or t+1 not in occ_a or t not in occ_b or t+1 not in occ_b:
                    continue
                cells_a_t = occ_a[t]
                cells_a_t1 = occ_a[t+1]
                cells_b_t = occ_b[t]
                cells_b_t1 = occ_b[t+1]
                if (cells_a_t & cells_b_t1) and (cells_a_t1 & cells_b_t):
                    conflicts.append({
                        'type': 'edge',
                        'time': t,
                        'pos': (path_a[t], path_a[t+1]),
                        'agents': [aid, bid]
                    })

    return conflicts


def assign_severity(conflict: Dict,
                    agents: List[AgentInstance],
                    priority_func: Callable[[AgentInstance], float]) -> float:
    """
    为单个冲突分配严重程度值。

    计算公式：severity = 冲突涉及智能体数量 * max(优先级)
    边冲突可能涉及两个智能体，顶点冲突可能涉及多个。

    :param conflict: 冲突字典
    :param agents: 所有智能体实例列表（用于根据ID查找智能体）
    :param priority_func: 输入智能体实例，返回优先级数值的函数
    :return: 严重程度值（浮点数）
    """
    agent_ids = conflict['agents']
    # 根据ID查找对应的智能体实例
    involved_agents = [a for a in agents if a.global_id in agent_ids]
    if not involved_agents:
        return 0.0

    # 计算最大优先级
    max_priority = max(priority_func(a) for a in involved_agents)
    severity = len(involved_agents) * max_priority

    # 根据冲突类型再调整，例如边冲突稍微加权
    if conflict['type'] == 'edge':
        severity *= 1.2

    return severity


def sort_conflicts(conflicts: List[Dict]) -> List[Dict]:
    """
    按严重程度降序排序冲突列表

    :param conflicts: 冲突列表（应包含 'severity' 键）
    :return: 排序后的列表
    """
    conflicts.sort(key=lambda c: c.get('severity', 0), reverse=True)
    return conflicts