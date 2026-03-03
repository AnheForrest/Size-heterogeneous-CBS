"""
CBS 高层模块
实现约束树搜索，解决多智能体路径冲突。
排序优先级：先最小化全局完成时间 (makespan)，再最小化总路径代价 (total_cost)，再最小化冲突数 (conflict_count)
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
        self.conflict_count = 0  # 新增：当前节点的冲突数量
        if paths:
            # makespan = 最大路径长度-1 (总时间步)
            # total_cost = 各智能体移动步数之和 (移动指位置变化，等待不计)
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
        self.open_set = []          # 优先队列，元素 (makespan, total_cost, conflict_count, node_id, node)
        self.best_makespan = float('inf')
        self.best_total_cost = float('inf')
        self.best_node = None
        self.expanded_signatures = set()   # 记录已扩展节点的签名，避免重复
        self.conflict_counter = {}          # 记录每个冲突被选择的次数，用于打破循环

    def _compute_metrics(self, paths: Dict[int, List]) -> Tuple[float, float]:
        """计算路径的 makespan 和 total_cost"""
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
        根据冲突生成子节点。
        对于顶点冲突，首先生成单约束子节点（每个智能体单独禁止），
        然后如果该冲突被选择的次数超过阈值（3次），则尝试生成一个同时禁止所有智能体的双约束子节点。
        :param node: 当前节点
        :param conflict: 冲突字典，包含 type, time, pos, agents
        :return: 子节点列表（已重规划，可能为空）
        """
        children = []
        ctype = conflict['type']
        t = conflict['time']
        agents_ids = conflict['agents']

        if ctype == 'vertex':
            pos = conflict['pos']
            # 生成单约束子节点
            for aid in agents_ids:
                print(f"      扩展顶点冲突子节点: 智能体 {aid} 禁止在 t={t} 位于 {pos}")
                child_node = node.apply_constraint(aid, (t, pos))
                new_path = self._replan(aid, child_node.constraints, node.paths)
                if new_path is not None:
                    new_paths = node.paths.copy()
                    new_paths[aid] = new_path
                    child_node.paths = new_paths
                    child_node.makespan, child_node.total_cost = self._compute_metrics(new_paths)
                    # 计算子节点的冲突数
                    child_node.conflict_count = self._compute_conflict_count(child_node)
                    children.append(child_node)
                    print(f"        重规划成功，新路径长度 {len(new_path)}，冲突数 {child_node.conflict_count}")
                    if len(new_path) > t:
                        print(f"        新路径在 t={t} 的位置: {new_path[t]}")
                else:
                    print(f"        重规划失败")

            # 获取该冲突的被选次数，如果超过阈值，尝试生成双约束子节点
            conflict_key = (ctype, t, pos, tuple(sorted(agents_ids)))
            count = self.conflict_counter.get(conflict_key, 0)
            print(f"        冲突计数器值: {count} (阈值3)")
            if count >= 3:  # 阈值设为3，可调整
                print(f"        冲突已被选择 {count} 次，尝试生成双约束子节点")
                # 同时禁止所有智能体
                double_node = node
                for aid in agents_ids:
                    double_node = double_node.apply_constraint(aid, (t, pos))
                # 重规划所有涉及的智能体
                new_paths = node.paths.copy()
                all_success = True
                for aid in agents_ids:
                    new_path = self._replan(aid, double_node.constraints, node.paths)
                    if new_path is None:
                        all_success = False
                        print(f"        智能体 {aid} 重规划失败，双约束子节点放弃")
                        break
                    new_paths[aid] = new_path
                if all_success:
                    double_node.paths = new_paths
                    double_node.makespan, double_node.total_cost = self._compute_metrics(new_paths)
                    double_node.conflict_count = self._compute_conflict_count(double_node)
                    children.append(double_node)
                    print(f"        双约束子节点生成成功，makespan={double_node.makespan}, total_cost={double_node.total_cost}，冲突数 {double_node.conflict_count}")
                else:
                    print(f"        双约束子节点生成失败")

        elif ctype == 'edge':
            if len(agents_ids) < 2:
                return children
            aid, bid = agents_ids[0], agents_ids[1]
            a_path = node.paths.get(aid)
            b_path = node.paths.get(bid)
            if a_path is None or b_path is None or t >= len(a_path) or t >= len(b_path):
                return children
            pos_a_t = a_path[t]
            pos_b_t = b_path[t]

            # 子节点1：禁止 a 在 t 时刻位于 pos_a_t
            print(f"      扩展边冲突子节点: 智能体 {aid} 禁止在 t={t} 位于 {pos_a_t}")
            child_a = node.apply_constraint(aid, (t, pos_a_t))
            new_path_a = self._replan(aid, child_a.constraints, node.paths)
            if new_path_a:
                new_paths_a = node.paths.copy()
                new_paths_a[aid] = new_path_a
                child_a.paths = new_paths_a
                child_a.makespan, child_a.total_cost = self._compute_metrics(new_paths_a)
                child_a.conflict_count = self._compute_conflict_count(child_a)
                children.append(child_a)
                print(f"        重规划成功，新路径长度 {len(new_path_a)}，冲突数 {child_a.conflict_count}")
                if len(new_path_a) > t:
                    print(f"        新路径在 t={t} 的位置: {new_path_a[t]}")
            else:
                print(f"        重规划失败")

            # 子节点2：禁止 b 在 t 时刻位于 pos_b_t
            print(f"      扩展边冲突子节点: 智能体 {bid} 禁止在 t={t} 位于 {pos_b_t}")
            child_b = node.apply_constraint(bid, (t, pos_b_t))
            new_path_b = self._replan(bid, child_b.constraints, node.paths)
            if new_path_b:
                new_paths_b = node.paths.copy()
                new_paths_b[bid] = new_path_b
                child_b.paths = new_paths_b
                child_b.makespan, child_b.total_cost = self._compute_metrics(new_paths_b)
                child_b.conflict_count = self._compute_conflict_count(child_b)
                children.append(child_b)
                print(f"        重规划成功，新路径长度 {len(new_path_b)}，冲突数 {child_b.conflict_count}")
                if len(new_path_b) > t:
                    print(f"        新路径在 t={t} 的位置: {new_path_b[t]}")
            else:
                print(f"        重规划失败")

        return children

    def _compute_conflict_count(self, node: CBSNode) -> int:
        """计算给定节点的路径集中的冲突数量"""
        # 先补齐路径至最大时间
        if not node.paths:
            return 0
        max_len = max(len(p) for p in node.paths.values())
        extended_paths = {}
        for aid, path in node.paths.items():
            if len(path) < max_len:
                last_pos = path[-1]
                extended = path + [last_pos] * (max_len - len(path))
                extended_paths[aid] = extended
            else:
                extended_paths[aid] = path
        temp_res = ReservationTable(bridge_cells=self.res_table.bridge_cells)
        for aid, path in extended_paths.items():
            agent = self.agent_dict[aid]
            temp_res.add(aid, path, agent.agent_class)
        conflicts = detect_conflicts(temp_res, self.agents, paths_override=extended_paths)
        return len(conflicts)

    def is_pruned(self, node: CBSNode) -> bool:
        """剪枝判断：基于最优解的双目标剪枝"""
        if node.makespan > self.best_makespan:
            return True
        if node.makespan == self.best_makespan and node.total_cost >= self.best_total_cost:
            return True
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

    def search(self, interactive: bool = False) -> Optional[Dict[int, List]]:
        """
        CBS 主循环。
        :param interactive: 是否交互式调试（暂停并显示路径）
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
        # 计算根节点的冲突数
        root_node.conflict_count = self._compute_conflict_count(root_node)

        # 优先队列元素：(makespan, total_cost, conflict_count, node_id, node)
        heapq.heappush(self.open_set, (root_node.makespan, root_node.total_cost, root_node.conflict_count, id(root_node), root_node))

        iteration = 0
        MAX_ITER = 500  # 增加最大迭代次数，防止过早终止
        while self.open_set:
            iteration += 1
            if iteration > MAX_ITER:
                print(f"达到最大迭代次数 {MAX_ITER}，可能陷入死锁，终止搜索。")
                return None

            _, _, _, _, node = heapq.heappop(self.open_set)
            print(f"\n===== 迭代 {iteration} =====")
            print(f"当前节点: makespan={node.makespan}, total_cost={node.total_cost}, conflict_count={node.conflict_count}")

            # 交互模式：打印当前所有智能体的路径
            if interactive:
                print("\n当前路径:")
                for agent in self.agents:
                    path = node.paths.get(agent.global_id, [])
                    if path:
                        start = path[0]
                        goal = path[-1]
                        length = len(path)-1
                        print(f"  智能体 {agent.id_str}: 起点 {start} -> 终点 {goal}, 步数 {length}, 路径前5步: {path[:5]}")
                    else:
                        print(f"  智能体 {agent.id_str}: 无路径")
                input("按回车键继续下一步（输入 q 退出）...")

            # 补齐路径至全局最大时间
            max_len = max(len(p) for p in node.paths.values())
            extended_paths = {}
            for aid, path in node.paths.items():
                if len(path) < max_len:
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

            # 检测冲突，传入延长后的路径
            conflicts = detect_conflicts(temp_res, self.agents, paths_override=extended_paths)
            print(f"检测到 {len(conflicts)} 个冲突")
            if not conflicts:
                self.update_best(node)
                print(f"找到无冲突解！迭代次数：{iteration}，makespan：{node.makespan}，total_cost：{node.total_cost}")
                return node.paths

            # 计算冲突严重程度并排序
            priority_func = lambda a: a.agent_class.width + a.agent_class.height
            for c in conflicts:
                c['severity'] = assign_severity(c, self.agents, priority_func)
            sorted_conflicts = sort_conflicts(conflicts)

            # 选择冲突：优先选择被选择次数较少的冲突，避免反复处理同一冲突
            chosen_conflict = None
            for c in sorted_conflicts:
                # 生成冲突唯一标识
                key = (c['type'], c['time'], c['pos'], tuple(sorted(c['agents'])))
                count = self.conflict_counter.get(key, 0)
                if count < 10:  # 阈值设为10，可调整
                    chosen_conflict = c
                    self.conflict_counter[key] = count + 1
                    break
            if chosen_conflict is None:
                # 如果所有冲突都被选择过多次，则强制选择第一个
                chosen_conflict = sorted_conflicts[0]
                key = (chosen_conflict['type'], chosen_conflict['time'], chosen_conflict['pos'], tuple(sorted(chosen_conflict['agents'])))
                self.conflict_counter[key] = self.conflict_counter.get(key, 0) + 1
                print("所有冲突均被选择多次，强制选择第一个。")

            print(f"选择冲突: 类型={chosen_conflict['type']}, 时间={chosen_conflict['time']}, "
                  f"位置={chosen_conflict['pos']}, 智能体={chosen_conflict['agents']}, severity={chosen_conflict['severity']:.2f}")

            # 扩展子节点
            children = self.expand_node(node, chosen_conflict)
            print(f"生成 {len(children)} 个子节点")
            for child in children:
                # 代价剪枝
                if self.is_pruned(child):
                    print(f"  子节点被代价剪枝 (makespan={child.makespan}, total_cost={child.total_cost})")
                    continue
                # 签名去重：将约束集转换为可哈希的字符串
                sig = str(sorted((aid, sorted(con_list)) for aid, con_list in child.constraints.items()))
                if sig in self.expanded_signatures:
                    print(f"  子节点签名重复，跳过")
                    continue
                self.expanded_signatures.add(sig)
                # 加入开放集，使用 (makespan, total_cost, conflict_count, id, node)
                heapq.heappush(self.open_set,
                               (child.makespan, child.total_cost, child.conflict_count, id(child), child))
                print(f"  子节点加入开放集: makespan={child.makespan}, total_cost={child.total_cost}, conflict_count={child.conflict_count}")

        print("CBS 搜索结束，无解。")
        return None