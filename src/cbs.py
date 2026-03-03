"""
CBS 高层模块
实现约束树搜索，解决多智能体路径冲突。
排序优先级：先最小化全局完成时间 (makespan)，再最小化总路径代价 (total_cost)
"""

import heapq
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from sh_agent import AgentInstance
from passable_graph import PassableGraph
from reservation_table import ReservationTable
from astar import astar
from conflict_detection import detect_conflicts, assign_severity, sort_conflicts


class CBSNode:
    """约束树节点"""

    def __init__(self, constraints: Dict[int, List[Tuple]], paths: Dict[int, List],
                 parent: 'CBSNode' = None):
        """
        :param constraints: 字典，key=agent_id, value=约束列表，每个约束为 (t, pos)
        :param paths: 字典，key=agent_id, value=路径列表（左上角坐标）
        :param parent: 父节点
        """
        self.constraints = constraints
        self.paths = paths
        self.parent = parent
        # 计算 metrics
        self.makespan = 0
        self.total_cost = 0
        if paths:
            lengths = [len(p)-1 for p in paths.values() if p]
            if lengths:
                self.total_cost = sum(lengths)
                self.makespan = max(lengths)

    def apply_constraint(self, agent_id: int, constraint: Tuple) -> 'CBSNode':
        """
        生成新节点，继承原有约束并添加一条新约束。
        :param agent_id: 智能体全局ID
        :param constraint: 约束 (t, pos)
        :return: 新节点（路径暂为空，需后续重规划）
        """
        new_constraints = deepcopy(self.constraints)
        if agent_id not in new_constraints:
            new_constraints[agent_id] = []
        new_constraints[agent_id].append(constraint)
        # 新节点路径为空，metrics 暂为 0，待重规划后更新
        return CBSNode(new_constraints, {}, parent=self)


class CBS:
    """CBS 算法主类"""

    def __init__(self, agents: List[AgentInstance],
                 passable_graphs: Dict[int, PassableGraph],
                 reservation_table: ReservationTable,
                 cr: List[List[int]]):
        """
        :param agents: 智能体实例列表
        :param passable_graphs: 类别 -> 可通行子图
        :param reservation_table: 全局预约表（用于软约束和桥信息）
        :param cr: 拥堵系数矩阵
        """
        self.agents = agents
        self.agent_dict = {a.global_id: a for a in agents}
        self.passable_graphs = passable_graphs
        self.res_table = reservation_table
        self.cr = cr
        self.open_set = []          # 优先队列，元素 (makespan, total_cost, node_id, node)
        self.best_makespan = float('inf')
        self.best_total_cost = float('inf')
        self.best_node = None
        # 用于TRDP剪枝的已访问状态（可选，暂不实现）
        self.closed_states = set()

    def _compute_metrics(self, paths: Dict[int, List]) -> Tuple[float, float]:
        """计算路径的 makespan 和 total_cost"""
        lengths = [len(p)-1 for p in paths.values() if p]
        if not lengths:
            return 0.0, 0.0
        return max(lengths), sum(lengths)

    def _replan(self, agent_id: int, constraints: Dict[int, List],
                current_paths: Dict[int, List]) -> Optional[List]:
        """为单个智能体重规划，考虑硬约束"""
        agent = self.agent_dict[agent_id]
        pg = self.passable_graphs[agent.agent_class.category]
        # 调用 A*，传入该智能体的约束列表
        path = astar(
            agent_instance=agent,
            passable_graph=pg,
            reservation_table=self.res_table,
            cr=self.cr,
            weight_cr=0.1,
            weight_bridge=10.0,
            weight_res=1.0,
            constraints=constraints.get(agent_id, [])
        )
        return path

    def expand_node(self, node: CBSNode, conflict: Dict) -> List[CBSNode]:
        """
        根据冲突生成两个子节点。
        :param node: 当前节点
        :param conflict: 冲突字典，包含 type, time, pos, agents
        :return: 子节点列表（已重规划，可能为空）
        """
        children = []
        ctype = conflict['type']
        t = conflict['time']
        agents_ids = conflict['agents']   # 至少两个智能体ID

        if ctype == 'vertex':
            pos = conflict['pos']
            for aid in agents_ids:
                # 生成新约束：禁止 aid 在时刻 t 位于 pos
                child_node = node.apply_constraint(aid, (t, pos))
                new_path = self._replan(aid, child_node.constraints, node.paths)
                if new_path is not None:
                    new_paths = node.paths.copy()
                    new_paths[aid] = new_path
                    child_node.paths = new_paths
                    # 更新 metrics
                    child_node.makespan, child_node.total_cost = self._compute_metrics(new_paths)
                    children.append(child_node)
                # 若重规划失败，丢弃该子节点

        elif ctype == 'edge':
            # 边冲突处理
            if len(agents_ids) < 2:
                return children
            aid, bid = agents_ids[0], agents_ids[1]
            a_path = node.paths.get(aid)
            b_path = node.paths.get(bid)
            if a_path is None or b_path is None or t >= len(a_path) or t >= len(b_path):
                return children
            pos_a_t = a_path[t]          # a 在 t 时刻的位置
            pos_b_t = b_path[t]          # b 在 t 时刻的位置

            # 子节点1：禁止 a 在 t 时刻位于 pos_a_t
            child_a = node.apply_constraint(aid, (t, pos_a_t))
            new_path_a = self._replan(aid, child_a.constraints, node.paths)
            if new_path_a:
                new_paths_a = node.paths.copy()
                new_paths_a[aid] = new_path_a
                child_a.paths = new_paths_a
                child_a.makespan, child_a.total_cost = self._compute_metrics(new_paths_a)
                children.append(child_a)

            # 子节点2：禁止 b 在 t 时刻位于 pos_b_t
            child_b = node.apply_constraint(bid, (t, pos_b_t))
            new_path_b = self._replan(bid, child_b.constraints, node.paths)
            if new_path_b:
                new_paths_b = node.paths.copy()
                new_paths_b[bid] = new_path_b
                child_b.paths = new_paths_b
                child_b.makespan, child_b.total_cost = self._compute_metrics(new_paths_b)
                children.append(child_b)

        return children

    def is_pruned(self, node: CBSNode) -> bool:
        """剪枝判断：基于最优解的双目标剪枝 + 可选TRDP"""
        # 代价剪枝：如果 makespan > best_makespan，直接剪枝
        if node.makespan > self.best_makespan:
            return True
        # 如果 makespan == best_makespan 且 total_cost >= best_total_cost，剪枝
        if node.makespan == self.best_makespan and node.total_cost >= self.best_total_cost:
            return True
        # 可以扩展 TRDP 剪枝，此处省略
        return False

    def update_best(self, node: CBSNode):
        """更新全局最优解"""
        if node.makespan < self.best_makespan:
            self.best_makespan = node.makespan
            self.best_total_cost = node.total_cost
            self.best_node = node
        elif node.makespan == self.best_makespan and node.total_cost < self.best_total_cost:
            self.best_total_cost = node.total_cost
            self.best_node = node

    def search(self) -> Optional[Dict[int, List]]:
        """
        CBS 主循环。
        :return: 无冲突的路径字典（key=agent_id, value=path），若无解返回 None
        """
        # 根节点：所有智能体独立规划最优路径（无约束）
        root_paths = {}
        root_constraints = {}
        for agent in self.agents:
            path = astar(
                agent_instance=agent,
                passable_graph=self.passable_graphs[agent.agent_class.category],
                reservation_table=self.res_table,
                cr=self.cr,
                weight_cr=0.1,
                weight_bridge=10.0,
                weight_res=1.0,
                constraints=[]
            )
            if path is None:
                print(f"警告：智能体 {agent.id_str} 初始规划无解，整个问题无解。")
                return None
            root_paths[agent.global_id] = path
        root_node = CBSNode(root_constraints, root_paths)

        # 将根节点加入优先队列
        heapq.heappush(self.open_set, (root_node.makespan, root_node.total_cost, id(root_node), root_node))

        iteration = 0
        while self.open_set:
            iteration += 1
            _, _, _, node = heapq.heappop(self.open_set)

            # 补齐路径至全局最大时间
            max_len = max(len(p) for p in node.paths.values())  # 路径长度（包含起点）
            extended_paths = {}
            for aid, path in node.paths.items():
                if len(path) < max_len:
                    # 复制最后一个位置补齐
                    last_pos = path[-1]
                    extended = path + [last_pos] * (max_len - len(path))
                    extended_paths[aid] = extended
                else:
                    extended_paths[aid] = path

            # 用延长后的路径构建临时预约表
            temp_res = ReservationTable(bridge_cells=self.res_table.bridge_cells)
            for aid, path in extended_paths.items():
                agent = self.agent_dict[aid]
                temp_res.add(aid, path, agent.agent_class)

            # 检测冲突
            conflicts = detect_conflicts(temp_res, self.agents)
            if not conflicts:
                # 找到无冲突解
                self.update_best(node)
                print(f"找到解，迭代次数：{iteration}，makespan：{node.makespan}，total_cost：{node.total_cost}")
                return node.paths

            # 计算冲突严重程度并排序
            priority_func = lambda a: a.agent_class.width + a.agent_class.height
            for c in conflicts:
                c['severity'] = assign_severity(c, self.agents, priority_func)
            sorted_conflicts = sort_conflicts(conflicts)
            # 选择最严重的冲突
            chosen_conflict = sorted_conflicts[0]

            # 扩展子节点
            children = self.expand_node(node, chosen_conflict)
            for child in children:
                if not self.is_pruned(child):
                    heapq.heappush(self.open_set,
                                   (child.makespan, child.total_cost, id(child), child))

        # 开放集耗尽，无解
        print("CBS 搜索结束，无解。")
        return None