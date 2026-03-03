# test_edge_conflict.py
from sh_agent import AgentClass, AgentInstance
from reservation_table import ReservationTable
from conflict_detection import detect_conflicts

# 创建两个智能体（全局ID 0 和 1）
cls = AgentClass(1, 1, 1)  # 1x1
a = AgentInstance(cls, 0, (0,0), (0,0))
a.global_id = 0
b = AgentInstance(cls, 1, (0,0), (0,0))
b.global_id = 1

# 手动设置路径，制造一个交换冲突
# 智能体0: 在 t=0 位于 (0,0)，t=1 位于 (1,0)
# 智能体1: 在 t=0 位于 (1,0)，t=1 位于 (0,0)
paths_override = {
    0: [(0,0), (1,0)],
    1: [(1,0), (0,0)]
}

# 创建一个空预约表（桥信息不重要）
res = ReservationTable()
# 添加路径到预约表（为了顶点冲突检测，但边冲突检测我们传入 paths_override 即可）
# 实际上 detect_conflicts 中边冲突不依赖预约表，只依赖 paths_override

agents = [a, b]
conflicts = detect_conflicts(res, agents, paths_override=paths_override)
print("检测到的冲突:")
for c in conflicts:
    print(c)