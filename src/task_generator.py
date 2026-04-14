"""
任务生成模块
为每类智能体随机生成指定数量的任务（起点和终点），并确保：
1. 每个任务的起点和终点在同一连通分量中（初步想法是用于无解检测，但是生成太多无解任务不利于测试）
2. 不同任务的起点/终点所占据的栅格互不重叠

！！！generate_tasks和retry_failed_tasks的核心区别是什么呢？仅仅是max_attempts?这种情况下真的有必要写两个方法吗？
"""

import random
from typing import List, Dict, Set, Tuple, Optional

from gridmap import GridMap
from sh_agent import AgentClass, AgentInstance
from passable_graph import PassableGraph

# 全局ID计数器，用于为每个智能体分配唯一整数ID
_next_global_id = 0

def generate_tasks(agent_classes: List[AgentClass],
                   counts: List[int],
                   grid_map: GridMap,  
                   passable_graphs: Dict[int, PassableGraph],
                   existing_occupied: Optional[Set[Tuple[int, int]]] = None,
                   max_attempts_per_agent: int = 1000) -> Tuple[List[AgentInstance], List[Tuple[int, int]]]:
    """
    为每类智能体生成指定数量的任务（起点和终点），并为每个智能体分配全局唯一ID。
    生成过程中会确保：
        - 起点和终点在对应可通行子图中连通
        - 不同智能体的起点/终点所占据的栅格不与 existing_occupied 中已有的栅格重叠
    如果某个智能体无法在 max_attempts_per_agent 次内找到合法任务，则跳过该智能体，
    并在返回的失败列表中记录 (category, instance_id)。

    :param agent_classes: 智能体类别列表
    :param counts: 每类智能体需要生成的数量，顺序与 agent_classes 一致
    :param grid_map: 基底地图
    :param passable_graphs: 字典，对应智能体类别和它的可通行图，键为类别编号，值为该类对应的 PassableGraph 对象
    :param existing_occupied: 已占用的栅格坐标集合（起点和终点的并集），若为 None 则初始化为空集
    :param max_attempts_per_agent: 每个智能体生成的最大尝试次数
    :return: (成功任务列表, 失败请求列表) 失败请求每个元素为 (category, instance_id)
    :raises ValueError: 如果某类没有合法位置，或参数长度不一致
    """
    global _next_global_id
    if len(agent_classes) != len(counts):
        raise ValueError("agent_classes 和 counts 长度必须一致")

    if existing_occupied is None:
        occupied_all: Set[Tuple[int, int]] = set()
    else:
        occupied_all = set(existing_occupied)  #复制一份，避免修改原集合

    tasks: List[AgentInstance] = []
    failed_requests: List[Tuple[int, int]] = []  # 每个元素 (category, instance_id)

    for cls, count in zip(agent_classes, counts):
        cat = cls.category
        pg = passable_graphs.get(cat)
        if pg is None:
            raise ValueError(f"类别 {cat} 没有对应的可通行子图")
        if not pg.V:
            raise ValueError(f"类别 {cat} 的可通行子图顶点集为空，无法生成任务")

        # 同类内实例编号从 0 开始
        for i in range(count):
            attempts = 0
            success = False
            while attempts < max_attempts_per_agent:
                # 随机选择起点和终点
                start = random.choice(pg.V)
                goal = random.choice(pg.V)
                while start == goal:   # 如果起点和终点相同，重新选择终点
                    goal = random.choice(pg.V)

                # 检查连通性
                if not pg.is_connected(start, goal):
                    attempts += 1
                    continue

                # 计算起点和终点占据的所有栅格
                start_cells = set(cls.get_occupied_cells(start))
                goal_cells = set(cls.get_occupied_cells(goal))

                # 检查是否与已占用的栅格冲突
                if (start_cells | goal_cells) & occupied_all:
                    attempts += 1
                    continue

                # 成功找到合法组合
                success = True
                break

            if not success:
                print(f"警告：类别 {cat} 的第 {i} 个智能体在尝试 {max_attempts_per_agent} 次后未能生成无冲突任务，已记录为失败")
                failed_requests.append((cat, i))
                continue

            # 创建智能体实例
            agent = AgentInstance(
                agent_class=cls,
                instance_id=i,
                start=start,
                goal=goal
            )
            agent.global_id = _next_global_id
            _next_global_id += 1
            tasks.append(agent)

            # 更新占用栅格集合
            occupied_all.update(start_cells)
            occupied_all.update(goal_cells)

    return tasks, failed_requests


def retry_failed_tasks(failed_requests: List[Tuple[int, int]],
                       agent_classes: List[AgentClass],
                       passable_graphs: Dict[int, PassableGraph],
                       existing_occupied: Set[Tuple[int, int]],
                       max_attempts: int = 10) -> List[AgentInstance]:
    """
    尝试重新生成失败的任务，要求新任务不与 existing_occupied 冲突。
    每个失败任务最多尝试 max_attempts 次，若成功则创建新智能体并返回。
    如果某个任务在尝试后仍失败，则打印警告并跳过。

    :param failed_requests: 失败请求列表，每个元素为 (category, instance_id)
    :param agent_classes: 智能体类别列表（用于获取类别对象）
    :param passable_graphs: 可通行子图字典
    :param existing_occupied: 已成功任务的起点/终点占用栅格集合（每个智能体起终点设定后会被更新）
    :param max_attempts: 每个失败任务的最大尝试次数
    :return: 新生成的成功任务列表
    """
    global _next_global_id
    # 构建类别到类别对象的映射，方便查找
    class_by_cat = {cls.category: cls for cls in agent_classes}
    new_tasks = []
    occupied_all = set(existing_occupied)  # 复制，避免修改原集合

    for (cat, instance_id) in failed_requests:
        cls = class_by_cat.get(cat)
        if cls is None:
            print(f"错误：类别 {cat} 不存在，跳过")
            continue
        pg = passable_graphs.get(cat)
        if pg is None:
            print(f"错误：类别 {cat} 无可通行子图，跳过")
            continue

        success = False
        for attempt in range(max_attempts):
            start = random.choice(pg.V)
            goal = random.choice(pg.V)
            if not pg.is_connected(start, goal):
                continue
            start_cells = set(cls.get_occupied_cells(start))
            goal_cells = set(cls.get_occupied_cells(goal))
            if (start_cells | goal_cells) & occupied_all:
                continue
            # 成功
            agent = AgentInstance(
                agent_class=cls,
                instance_id=instance_id,
                start=start,
                goal=goal
            )
            agent.global_id = _next_global_id
            _next_global_id += 1
            new_tasks.append(agent)
            occupied_all.update(start_cells)
            occupied_all.update(goal_cells)
            success = True
            break

        if not success:
            print(f"警告：类别 {cat} 的第 {instance_id} 个智能体在重试 {max_attempts} 次后仍无法生成无冲突任务，已永久跳过")

    # 更新原始 existing_occupied 集合（通过传入的可变集合直接修改）
    existing_occupied.update(occupied_all - set(existing_occupied))

    return new_tasks


def filter_feasible_tasks(tasks: List[AgentInstance],
                          passable_graphs: Dict[int, PassableGraph]) -> List[AgentInstance]:
    """
    剔除起点到终点不可达的任务，并打印警告信息。
    
    """
    feasible = []
    for task in tasks:
        cat = task.agent_class.category
        pg = passable_graphs.get(cat)
        if pg is None:
            print(f"警告：智能体 {task.id_str}（类别 {cat}）无可通行子图，已剔除")
            continue
        if pg.is_connected(task.start, task.goal):
            feasible.append(task)
        else:
            print(f"警告：智能体 {task.id_str}（类别 {cat}）起点 {task.start} 到终点 {task.goal} 不可达，已剔除")
    return feasible