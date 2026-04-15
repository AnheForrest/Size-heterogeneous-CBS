"""
cbs_hcbs_base.py
HCBS-base：仅增加可通行子图保证物理可行性，无任何启发式优化。
修复：使用 ReservationTable 和 detect_conflicts 进行真实的二维冲突检测。
"""

import heapq
import time
from typing import Dict, List, Optional, Tuple

from sh_agent import AgentInstance
from passable_graph import PassableGraph
from reservation_table import ReservationTable
from astar import astar
from conflict_detection import detect_conflicts


class CBSNode:
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
        from copy import deepcopy
        new_constraints = deepcopy(self.constraints)
        if agent_id not in new_constraints:
            new_constraints[agent_id] = []
        new_constraints[agent_id].append(constraint)
        return CBSNode(new_constraints, {}, parent=self)


class HCBSBase:
    """HCBS-base 求解器：仅可通行子图，无启发式优化"""

    def __init__(self, agents: List[AgentInstance],
                 passable_graphs: Dict[int, PassableGraph],
                 reservation_table: ReservationTable,
                 cr: List[List[int]]):
        self.agents = agents
        self.agent_dict = {a.global_id: a for a in agents}
        self.passable_graphs = passable_graphs
        self.res_table = reservation_table
        self.cr = cr
        self.open_set = []
        self.best_makespan = float('inf')
        self.best_total_cost = float('inf')
        self.best_node = None
        self.time_limit = 60.0
        self.expanded_nodes = 0

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

    def _replan(self, agent_id: int, constraints: Dict[int, List]) -> Optional[List]:
        agent = self.agent_dict[agent_id]
        pg = self.passable_graphs[agent.agent_class.category]
        path = astar(
            agent_instance=agent,
            passable_graph=pg,
            reservation_table=self.res_table,
            cr=self.cr,
            weight_cr=0.0,
            weight_bridge=0.0,
            weight_res=0.0,
            constraints=constraints.get(agent_id, [])
        )
        return path

    def _detect_conflicts(self, paths: Dict[int, List]) -> List[Dict]:
        """使用真实的二维冲突检测"""
        if not paths:
            return []
        max_len = max(len(p) for p in paths.values())
        extended_paths = {}
        for aid, path in paths.items():
            if len(path) < max_len:
                last = path[-1]
                extended_paths[aid] = path + [last] * (max_len - len(path))
            else:
                extended_paths[aid] = path[:max_len]

        temp_res = ReservationTable(bridge_cells=self.res_table.bridge_cells)
        for aid, path in extended_paths.items():
            agent = self.agent_dict[aid]
            temp_res.add(aid, path, agent.agent_class)

        conflicts = detect_conflicts(temp_res, self.agents, paths_override=extended_paths)
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

    def search(self, interactive: bool = False) -> Tuple[bool, Optional[Dict[int, List]], Dict]:
        start_time = time.time()
        root_paths = {}
        for agent in self.agents:
            path = self._replan(agent.global_id, {})
            if path is None:
                return False, None, {'reason': 'no_solution', 'nodes_expanded': 0}
            root_paths[agent.global_id] = path
        root_node = CBSNode({}, root_paths)
        heapq.heappush(self.open_set, (root_node.makespan, root_node.total_cost, id(root_node), root_node))

        while self.open_set:
            if time.time() - start_time > self.time_limit:
                return False, None, {'reason': 'timeout', 'nodes_expanded': self.expanded_nodes}
            _, _, _, node = heapq.heappop(self.open_set)
            self.expanded_nodes += 1

            conflicts = self._detect_conflicts(node.paths)
            if not conflicts:
                self._update_best(node)
                return True, node.paths, {
                    'reason': 'success',
                    'nodes_expanded': self.expanded_nodes,
                    'makespan': node.makespan,
                    'cost': node.total_cost
                }

            conflict = conflicts[0]
            children = []
            if conflict['type'] == 'vertex':
                t = conflict['time']
                pos = conflict['pos']
                for aid in conflict['agents']:
                    child = node.apply_constraint(aid, (t, pos))
                    new_path = self._replan(aid, child.constraints)
                    if new_path:
                        new_paths = node.paths.copy()
                        new_paths[aid] = new_path
                        child.paths = new_paths
                        child.makespan, child.total_cost = self._compute_metrics(new_paths)
                        children.append(child)
            else:  # edge
                t = conflict['time']
                aid1, aid2 = conflict['agents'][0], conflict['agents'][1]
                pos1, pos2 = node.paths[aid1][t], node.paths[aid2][t]
                for aid, pos in [(aid1, pos1), (aid2, pos2)]:
                    child = node.apply_constraint(aid, (t, pos))
                    new_path = self._replan(aid, child.constraints)
                    if new_path:
                        new_paths = node.paths.copy()
                        new_paths[aid] = new_path
                        child.paths = new_paths
                        child.makespan, child.total_cost = self._compute_metrics(new_paths)
                        children.append(child)

            for child in children:
                if self._is_pruned(child):
                    continue
                heapq.heappush(self.open_set, (child.makespan, child.total_cost, id(child), child))

        return False, None, {'reason': 'no_solution', 'nodes_expanded': self.expanded_nodes}