"""
cbs_classic.py
经典 CBS 算法（质点模型）
- 所有智能体视为 1×1，在基底地图的可行栅格上移动
- 无任何软约束（拥堵系数、桥栅格、预约表）
- 无优先级排序、无拥堵感知冲突选择、无双约束扩展、无 TRDP 去重
- 仅保留代价剪枝
"""

import heapq
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from gridmap import GridMap
from sh_agent import AgentInstance


class CBSNode:
    """约束树节点"""

    def __init__(self, constraints: Dict[int, List[Tuple]], paths: Dict[int, List],
                 parent: 'CBSNode' = None):
        self.constraints = constraints
        self.paths = paths
        self.parent = parent
        self.makespan = 0
        self.total_cost = 0
        if paths:
            max_len = 0
            total_moves = 0
            for path in paths.values():
                if not path:
                    continue
                steps = len(path) - 1
                if steps > max_len:
                    max_len = steps
                moves = 0
                for i in range(1, len(path)):
                    if path[i] != path[i-1]:
                        moves += 1
                total_moves += moves
            self.makespan = max_len
            self.total_cost = total_moves

    def apply_constraint(self, agent_id: int, constraint: Tuple) -> 'CBSNode':
        new_constraints = deepcopy(self.constraints)
        if agent_id not in new_constraints:
            new_constraints[agent_id] = []
        new_constraints[agent_id].append(constraint)
        return CBSNode(new_constraints, {}, parent=self)


class ClassicCBS:
    """经典 CBS 求解器（质点模型）"""

    def __init__(self, agents: List[AgentInstance], grid_map: GridMap):
        """
        :param agents: 智能体实例列表（尺寸信息将被忽略，统一视为1×1）
        :param grid_map: 基底栅格地图
        """
        self.agents = agents
        self.agent_dict = {a.global_id: a for a in agents}
        self.grid_map = grid_map
        self.open_set = []
        self.best_makespan = float('inf')
        self.best_total_cost = float('inf')
        self.best_node = None
        self.expanded_signatures = set()
        self.time_limit = 60.0

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """曼哈顿距离"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """四向移动邻居，仅返回可通行栅格"""
        x, y = pos
        candidates = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        return [p for p in candidates if self.grid_map.is_passable(p[0], p[1])]

    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int],
               constraints: List[Tuple[int, Tuple[int, int]]]) -> Optional[List[Tuple[int, int]]]:
        """
        经典 A*：质点，仅遵守硬约束
        :param constraints: 列表，元素为 (t, pos)
        """
        constraint_dict = {}
        for t, pos in constraints:
            if t not in constraint_dict:
                constraint_dict[t] = set()
            constraint_dict[t].add(pos)

        t0 = 0
        if t0 in constraint_dict and start in constraint_dict[t0]:
            return None

        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, goal), 0, t0, start))
        g_score = {(start, t0): 0}
        came_from = {}

        while open_set:
            f, g, t, pos = heapq.heappop(open_set)
            if pos == goal:
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
            for nbr in self._get_neighbors(pos):
                if nt in constraint_dict and nbr in constraint_dict[nt]:
                    continue
                ng = g + 1
                state = (nbr, nt)
                if ng < g_score.get(state, float('inf')):
                    g_score[state] = ng
                    f_val = ng + self._heuristic(nbr, goal)
                    heapq.heappush(open_set, (f_val, ng, nt, nbr))
                    came_from[state] = (pos, t)
            # 等待
            if nt not in constraint_dict or pos not in constraint_dict[nt]:
                ng = g + 1
                state = (pos, nt)
                if ng < g_score.get(state, float('inf')):
                    g_score[state] = ng
                    f_val = ng + self._heuristic(pos, goal)
                    heapq.heappush(open_set, (f_val, ng, nt, pos))
                    came_from[state] = (pos, t)
        return None

    def _compute_metrics(self, paths: Dict[int, List]) -> Tuple[int, int]:
        max_len = 0
        total_moves = 0
        for path in paths.values():
            if not path:
                continue
            steps = len(path) - 1
            if steps > max_len:
                max_len = steps
            moves = 0
            for i in range(1, len(path)):
                if path[i] != path[i-1]:
                    moves += 1
            total_moves += moves
        return max_len, total_moves

    def _detect_conflicts(self, paths: Dict[int, List]) -> List[Dict]:
        """检测顶点冲突和边冲突"""
        conflicts = []
        max_len = max(len(p) for p in paths.values()) if paths else 0
        if max_len == 0:
            return conflicts
        # 补齐路径
        extended = {}
        for aid, path in paths.items():
            if len(path) < max_len:
                last = path[-1]
                extended[aid] = path + [last] * (max_len - len(path))
            else:
                extended[aid] = path

        # 顶点冲突
        for t in range(max_len):
            pos_to_agents = {}
            for aid, path in extended.items():
                pos = path[t]
                pos_to_agents.setdefault(pos, []).append(aid)
            for pos, agents in pos_to_agents.items():
                if len(agents) > 1:
                    conflicts.append({
                        'type': 'vertex',
                        'time': t,
                        'pos': pos,
                        'agents': agents
                    })

        # 边冲突
        for t in range(max_len - 1):
            for i, aid1 in enumerate(self.agent_dict.keys()):
                for aid2 in list(self.agent_dict.keys())[i+1:]:
                    p1 = extended.get(aid1)
                    p2 = extended.get(aid2)
                    if not p1 or not p2 or t >= len(p1) or t >= len(p2):
                        continue
                    if p1[t] == p2[t+1] and p1[t+1] == p2[t]:
                        conflicts.append({
                            'type': 'edge',
                            'time': t,
                            'pos': (p1[t], p1[t+1]),
                            'agents': [aid1, aid2]
                        })
        return conflicts

    def _is_pruned(self, node: CBSNode) -> bool:
        if node.makespan > self.best_makespan:
            return True
        if node.makespan == self.best_makespan and node.total_cost >= self.best_total_cost:
            return True
        return False

    def _update_best(self, node: CBSNode):
        if node.makespan < self.best_makespan:
            self.best_makespan = node.makespan
            self.best_total_cost = node.total_cost
            self.best_node = node
        elif node.makespan == self.best_makespan and node.total_cost < self.best_total_cost:
            self.best_total_cost = node.total_cost
            self.best_node = node

    def search(self) -> Tuple[bool, Optional[Dict[int, List]], Dict]:
        """
        经典 CBS 主循环
        :return: (success, paths, stats)
        """
        start_time = time.time()
        root_paths = {}
        for agent in self.agents:
            path = self._astar(agent.start, agent.goal, [])
            if path is None:
                print(f"警告：智能体 {agent.id_str} 初始规划无解。")
                return False, None, {'reason': 'no_solution', 'nodes_expanded': 0}
            root_paths[agent.global_id] = path
        root_node = CBSNode({}, root_paths)
        heapq.heappush(self.open_set, (root_node.makespan, root_node.total_cost, id(root_node), root_node))

        iteration = 0
        while self.open_set:
            if time.time() - start_time > self.time_limit:
                return False, None, {'reason': 'timeout', 'nodes_expanded': len(self.expanded_signatures)}
            iteration += 1
            _, _, _, node = heapq.heappop(self.open_set)

            conflicts = self._detect_conflicts(node.paths)
            if not conflicts:
                self._update_best(node)
                return True, node.paths, {
                    'reason': 'success',
                    'nodes_expanded': len(self.expanded_signatures),
                    'makespan': node.makespan,
                    'cost': node.total_cost
                }

            # 随机选择一个冲突（经典 CBS 无启发式选择）
            conflict = conflicts[0]

            children = []
            if conflict['type'] == 'vertex':
                t = conflict['time']
                pos = conflict['pos']
                for aid in conflict['agents']:
                    child = node.apply_constraint(aid, (t, pos))
                    new_path = self._astar(
                        self.agent_dict[aid].start,
                        self.agent_dict[aid].goal,
                        child.constraints.get(aid, [])
                    )
                    if new_path:
                        new_paths = node.paths.copy()
                        new_paths[aid] = new_path
                        child.paths = new_paths
                        child.makespan, child.total_cost = self._compute_metrics(new_paths)
                        children.append(child)
            elif conflict['type'] == 'edge':
                t = conflict['time']
                aid1, aid2 = conflict['agents'][0], conflict['agents'][1]
                pos1, pos2 = node.paths[aid1][t], node.paths[aid2][t]
                # 约束 aid1
                child1 = node.apply_constraint(aid1, (t, pos1))
                new_path1 = self._astar(
                    self.agent_dict[aid1].start,
                    self.agent_dict[aid1].goal,
                    child1.constraints.get(aid1, [])
                )
                if new_path1:
                    new_paths1 = node.paths.copy()
                    new_paths1[aid1] = new_path1
                    child1.paths = new_paths1
                    child1.makespan, child1.total_cost = self._compute_metrics(new_paths1)
                    children.append(child1)
                # 约束 aid2
                child2 = node.apply_constraint(aid2, (t, pos2))
                new_path2 = self._astar(
                    self.agent_dict[aid2].start,
                    self.agent_dict[aid2].goal,
                    child2.constraints.get(aid2, [])
                )
                if new_path2:
                    new_paths2 = node.paths.copy()
                    new_paths2[aid2] = new_path2
                    child2.paths = new_paths2
                    child2.makespan, child2.total_cost = self._compute_metrics(new_paths2)
                    children.append(child2)

            for child in children:
                if self._is_pruned(child):
                    continue
                sig = str(sorted((aid, sorted(child.constraints.get(aid, []))) for aid in child.constraints))
                if sig in self.expanded_signatures:
                    continue
                self.expanded_signatures.add(sig)
                heapq.heappush(self.open_set, (child.makespan, child.total_cost, id(child), child))

        return False, None, {'reason': 'no_solution', 'nodes_expanded': len(self.expanded_signatures)}
