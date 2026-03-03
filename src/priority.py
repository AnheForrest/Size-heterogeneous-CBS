"""
优先级排序模块
提供基于智能体尺寸和曼哈顿距离的优先级计算与任务排序功能。
"""

from typing import List

from sh_agent import AgentClass, AgentInstance


def compute_priority(agent_class: AgentClass) -> int:
    """
    根据智能体类别计算尺寸优先级。
    优先级定义为智能体尺寸的宽度与高度之和 (width + height)。
    该值越大，表示智能体单次移动影响的栅格越多，应优先规划。

    :param agent_class: 智能体类别对象
    :return: 尺寸优先级值（整数）
    """
    return agent_class.width + agent_class.height


def manhattan_distance(start: tuple, goal: tuple) -> int:
    """
    计算起点与终点之间的曼哈顿距离。
    :param start: 起点左上角坐标 (x, y)
    :param goal: 终点左上角坐标 (x, y)
    :return: 曼哈顿距离
    """
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def sort_tasks_by_priority(tasks: List[AgentInstance]) -> List[AgentInstance]:
    """
    按优先级降序对任务列表排序，返回新的排序后列表。
    排序规则：
        - 首先按尺寸优先级（width+height）降序。
        - 若尺寸优先级相同，则按起点到终点的曼哈顿距离升序（距离近的优先）。
    使用稳定排序，保持相同优先级键的任务相对顺序不变。

    :param tasks: AgentInstance 对象列表
    :return: 排序后的新列表
    """
    return sorted(
        tasks,
        key=lambda t: (
            -compute_priority(t.agent_class),           # 主键：尺寸优先级降序
            manhattan_distance(t.start, t.goal)         # 次键：曼哈顿距离升序
        )
    )