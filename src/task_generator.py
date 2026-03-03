"""
任务生成模块
为每类智能体随机生成指定数量的任务（起点和终点），并提供可达性过滤功能。
"""

import random
from typing import List, Dict

from gridmap import GridMap
from sh_agent import AgentClass, AgentInstance
from passable_graph import PassableGraph


def generate_tasks(agent_classes: List[AgentClass],
                   counts: List[int],
                   grid_map: GridMap,
                   passable_graphs: Dict[int, PassableGraph]) -> List[AgentInstance]:
    """
    为每类智能体生成指定数量的任务（起点和终点）。

    :param agent_classes: 智能体类别列表
    :param counts: 每类智能体需要生成的数量，顺序与 agent_classes 一致
    :param grid_map: 基底地图（用于可能的边界检查，但此处主要依赖可通行子图）
    :param passable_graphs: 字典，键为类别编号，值为该类对应的 PassableGraph 对象
    :return: AgentInstance 对象列表
    :raises ValueError: 如果某类没有合法位置，或参数长度不一致
    """
    if len(agent_classes) != len(counts):
        raise ValueError("agent_classes 和 counts 长度必须一致")

    tasks = []

    for cls, count in zip(agent_classes, counts):
        cat = cls.category
        pg = passable_graphs.get(cat)
        if pg is None:
            raise ValueError(f"类别 {cat} 没有对应的可通行子图")

        if not pg.V:
            raise ValueError(f"类别 {cat} 的可通行子图顶点集为空，无法生成任务")

        # 同类内实例编号从 0 开始
        for i in range(count):
            # 从可通行子图的合法顶点集中随机选取起点和终点
            start = random.choice(pg.V)
            goal = random.choice(pg.V)

            # 可选：确保起点和终点不同（可注释掉）
            # while start == goal:
            #     goal = random.choice(pg.V)

            agent = AgentInstance(
                agent_class=cls,
                instance_id=i,          # 同类内唯一编号
                start=start,
                goal=goal
            )
            tasks.append(agent)

    return tasks


def filter_feasible_tasks(tasks: List[AgentInstance],
                          passable_graphs: Dict[int, PassableGraph]) -> List[AgentInstance]:
    """
    剔除起点到终点不可达的任务，并打印警告信息。

    :param tasks: AgentInstance 对象列表
    :param passable_graphs: 类别到 PassableGraph 的映射
    :return: 过滤后的可行任务列表
    """
    feasible = []
    for task in tasks:
        cat = task.agent_class.category
        pg = passable_graphs.get(cat)
        if pg is None:
            print(f"警告：智能体 {task.id}（类别 {cat}）无可通行子图，已剔除")
            continue
        if pg.is_connected(task.start, task.goal):
            feasible.append(task)
        else:
            print(f"警告：智能体 {task.id}（类别 {cat}）起点 {task.start} 到终点 {task.goal} 不可达，已剔除")
    return feasible