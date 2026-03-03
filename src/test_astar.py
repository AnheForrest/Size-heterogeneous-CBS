# test_astar.py
import random
from gridmap import GridMap
from sh_agent import AgentClass, AgentInstance
from passable_graph import PassableGraph
from astar import astar

def main():
    # 创建一个无障碍的 5x5 地图
    grid = GridMap(5, 5)
    # 不生成障碍物（或者生成0%）
    # grid.create_random_obstacles(0.0)  # 如果需要障碍，可以取消注释

    # 定义一个 1x1 智能体类别
    cls = AgentClass(category=1, width=1, height=1)
    pg = PassableGraph(cls.category)
    pg.build_from(grid, cls)
    pg.width = cls.width
    pg.height = cls.height
    pg.find_bridges(cls)  # 桥检测，但无障碍地图不会有桥

    # 创建智能体实例：起点 (0,0)，终点 (4,4)（曼哈顿距离 8）
    agent = AgentInstance(agent_class=cls, instance_id=0, start=(0,0), goal=(4,4))

    # 计算拥堵系数（所有栅格拥堵系数应为0，因为只有一类智能体）
    # 这里简化，直接传入全0矩阵
    cr = [[0]*5 for _ in range(5)]

    # 创建空预约表
    from reservation_table import ReservationTable
    res = ReservationTable()

    # 调用 A*，惩罚权重全设为0，只考虑步数
    path = astar(
        agent_instance=agent,
        passable_graph=pg,
        reservation_table=res,
        cr=cr,
        weight_cr=0.0,
        weight_bridge=0.0,
        weight_res=0.0,
        constraints=[]
    )

    print("规划路径:", path)
    print("路径长度（含起点）:", len(path))
    print("曼哈顿距离:", abs(agent.start[0]-agent.goal[0]) + abs(agent.start[1]-agent.goal[1]))
    if path:
        # 检查路径是否最短（长度-1 应等于曼哈顿距离）
        print("步数:", len(path)-1)
    else:
        print("无路径")

if __name__ == "__main__":
    main()