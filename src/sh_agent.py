"""
异构智能体模块
功能：
1.输入长宽定义智能体类 ->__init__
2.定义同一智能体类的四向移动和等待
3.定义具体任务智能体
"""


from typing import List, Tuple, Optional

class AgentClass:
    """
    同类智能体的模板，同一类别的智能体具有相同的尺寸和移动规则。
    """
    def __init__(self, category: int, width: int, height: int):
        """
        初始化智能体类别。

        :param category: 类别编号
        :param width: 智能体宽度（占据的栅格列数）
        :param height: 智能体高度（占据的栅格行数）
        """
        self.category = category
        self.width = width
        self.height = height

    def move_up(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """上移一步：y 坐标减 1"""
        x, y = pos
        return (x, y - 1)

    def move_down(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """下移一步：y 坐标加 1"""
        x, y = pos
        return (x, y + 1)

    def move_left(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """左移一步：x 坐标减 1"""
        x, y = pos
        return (x - 1, y)

    def move_right(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """右移一步：x 坐标加 1"""
        x, y = pos
        return (x + 1, y)

    def wait(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """等待：位置不变"""
        return pos

    def get_occupied_cells(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        根据当前左上角位置，返回智能体占用的所有栅格坐标列表。

        :param pos: 左上角坐标 (x, y)
        :return: 占用栅格的坐标列表，每个元素为 (x, y)
        """
        x, y = pos
        cells = []
        for dx in range(self.width):
            for dy in range(self.height):
                cells.append((x + dx, y + dy))
        return cells

    def __repr__(self) -> str:
        return f"AgentClass(cat={self.category}, size=({self.width},{self.height}))"


class AgentInstance:
    """
    具体任务智能体，绑定一个智能体类别，并具有起点、终点、路径等信息。
    """
    def __init__(self, agent_class: AgentClass, instance_id: int,
                 start: Tuple[int, int], goal: Tuple[int, int]):
        """
        初始化任务智能体。

        :param agent_class: 所属的智能体类别
        :param instance_id: 实例编号（同类内唯一）
        :param start: 起点左上角坐标 (x, y)
        :param goal: 终点左上角坐标 (x, y)
        """
        self.agent_class = agent_class
        self.id = instance_id
        self.start = start
        self.goal = goal
        self.path: Optional[List[Tuple[int, int]]] = None   # 规划路径，每个元素为位置坐标
        self.arrival_time: Optional[int] = None             # 到达目标点的时间步

    def set_path(self, path: List[Tuple[int, int]]) -> None:
        """设置智能体的规划路径"""
        self.path = path
        if path:
            self.arrival_time = len(path) - 1   # 假设路径长度即为到达时间（等待也算时间步）
        else:
            self.arrival_time = None

    def get_path(self) -> Optional[List[Tuple[int, int]]]:
        """获取智能体的规划路径"""
        return self.path

    def __repr__(self) -> str:
        return (f"AgentInstance(class={self.agent_class.category}, id={self.id}, "
                f"start={self.start}, goal={self.goal})")