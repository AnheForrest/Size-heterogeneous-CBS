"""
debug_conflict.py
针对特定种子详细调试冲突检测
"""
import random
import numpy as np
from gridmap import GridMap
from passable_graph import PassableGraph
from task_generator import generate_tasks
from cbs_hcbs_base import HCBSBase
from reservation_table import ReservationTable
from sh_agent import AgentClass
from congestion_coefficient import compute_cr
from conflict_detection import detect_conflicts

SEED = 4096
MAP_W, MAP_H = 20, 20
OBSTACLE = 0.2
N = 6
SCENE_CONFIGS = [
    {'w': 1, 'h': 1, 'ratio': 0.6},
    {'w': 1, 'h': 2, 'ratio': 0.2},
    {'w': 2, 'h': 1, 'ratio': 0.2}
]

random.seed(SEED)
np.random.seed(SEED)

g_map = GridMap(MAP_W, MAP_H)
g_map.create_random_obstacles(OBSTACLE)

agent_classes, counts = [], []
curr = 0
for cat_id, cfg in enumerate(SCENE_CONFIGS):
    cnt = int(N * cfg['ratio'])
    cls = AgentClass(cat_id, cfg['w'], cfg['h'])
    agent_classes.append(cls)
    counts.append(cnt)
    curr += cnt
    cat_id += 1
if curr < N:
    counts[-1] += (N - curr)

passable_dict = {}
for cls in agent_classes:
    pg = PassableGraph(cls.category)
    pg.build_from(g_map, cls)
    pg.width, pg.height = cls.width, cls.height
    passable_dict[cls.category] = pg

agents, _ = generate_tasks(
    agent_classes, counts, g_map, passable_dict,
    existing_occupied=None, max_attempts_per_agent=1500
)

all_bridges = set()
for pg in passable_dict.values():
    pg.find_bridges(agent_classes[0])
    all_bridges.update(pg.bridges)
res_tab = ReservationTable(bridge_cells=all_bridges)
cr = compute_cr(g_map, passable_dict)

solver = HCBSBase(agents, passable_dict, res_tab, cr)
solver.time_limit = 5.0
success, paths, stats = solver.search()

if not success:
    print("未成功")
    exit()

# 补齐路径
max_len = max(len(p) for p in paths.values())
extended = {}
for aid, path in paths.items():
    if len(path) < max_len:
        extended[aid] = path + [path[-1]] * (max_len - len(path))
    else:
        extended[aid] = path[:max_len]

# 手动检查一个已知冲突时刻（根据你的截图，例如 t=19）
t = 19
print(f"\n===== 时刻 {t} 智能体占用详情 =====")
for agent in agents:
    aid = agent.global_id
    pos = extended[aid][t]
    cells = agent.agent_class.get_occupied_cells(pos)
    print(f"{agent.id_str} ({agent.agent_class.width}x{agent.agent_class.height}) 左上角 {pos} 占用: {cells}")

# 检查是否有重叠
all_cells = {}
for agent in agents:
    for cell in agent.agent_class.get_occupied_cells(extended[agent.global_id][t]):
        if cell in all_cells:
            print(f"\n!!! 冲突: 栅格 {cell} 同时被 {all_cells[cell]} 和 {agent.id_str} 占用")
        else:
            all_cells[cell] = agent.id_str

# 调用 detect_conflicts 看它是否报告
temp_res = ReservationTable(bridge_cells=all_bridges)
for aid, path in extended.items():
    ag = next(a for a in agents if a.global_id == aid)
    temp_res.add(aid, path, ag.agent_class)
conflicts = detect_conflicts(temp_res, agents, paths_override=extended)
print(f"\ndetect_conflicts 共发现 {len(conflicts)} 个冲突")
for c in conflicts:
    if c['time'] == t:
        print(f"时刻 {t} 冲突详情: {c}")