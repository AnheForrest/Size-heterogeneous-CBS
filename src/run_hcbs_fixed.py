"""
run_hcbs_fixed.py
仅运行修复后的 HCBS-base，获取其在困难场景下的真实成功率
场景：长条占比 80%，N=5，20×20，障碍率 0.2
重复 20 次（快速验证）
"""

import os, sys, time, random, csv
import numpy as np
from datetime import datetime

from gridmap import GridMap
from passable_graph import PassableGraph
from task_generator import generate_tasks
from cbs_hcbs_base import HCBSBase
from reservation_table import ReservationTable
from sh_agent import AgentClass
from congestion_coefficient import compute_cr

TIME_LIMIT = 5.0
MAX_NODES = 3000
REPEAT = 20
MAP_W, MAP_H = 20, 20
OBSTACLE = 0.2
N = 5

SCENE = [
    {'w': 1, 'h': 1, 'ratio': 0.2},
    {'w': 1, 'h': 2, 'ratio': 0.4},
    {'w': 2, 'h': 1, 'ratio': 0.4}
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("..", "test", f"hcbs_fixed_{timestamp}")
os.makedirs(OUT_DIR, exist_ok=True)

def run_trial(seed):
    random.seed(seed)
    np.random.seed(seed)
    start = time.time()
    try:
        g_map = GridMap(MAP_W, MAP_H)
        g_map.create_random_obstacles(OBSTACLE)

        agent_classes, counts = [], []
        curr = 0
        for cid, cfg in enumerate(SCENE):
            cnt = int(N * cfg['ratio'])
            cls = AgentClass(cid, cfg['w'], cfg['h'])
            agent_classes.append(cls)
            counts.append(cnt)
            curr += cnt
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
        if len(agents) < int(N * 0.8):
            raise Exception(f"Task gen failed: {len(agents)} agents")

        all_bridges = set()
        for pg in passable_dict.values():
            pg.find_bridges(agent_classes[0])
            all_bridges.update(pg.bridges)
        res_tab = ReservationTable(bridge_cells=all_bridges)
        cr = compute_cr(g_map, passable_dict)

        solver = HCBSBase(agents, passable_dict, res_tab, cr)
        solver.time_limit = TIME_LIMIT
        success, paths, stats = solver.search(interactive=False)

        elapsed = time.time() - start
        nodes = stats.get('nodes_expanded', 0) if stats else 0
        if elapsed > TIME_LIMIT or nodes > MAX_NODES:
            success = False
            if stats:
                stats['reason'] = 'timeout' if elapsed > TIME_LIMIT else 'node_limit'

        return {
            'seed': seed, 'success': success, 'time': elapsed,
            'makespan': stats.get('makespan', -1) if stats and success else -1,
            'nodes': nodes
        }
    except Exception as e:
        return {
            'seed': seed, 'success': False, 'time': time.time()-start,
            'makespan': -1, 'nodes': 0, 'error': str(e)
        }

print(f"运行修复后的 HCBS-base，场景：长条占比80%，N=5，重复{REPEAT}次")
results = []
succ = 0
times = []
for i in range(REPEAT):
    seed = hash(f"hcbs_fixed_{i}") % 100000 + i
    res = run_trial(seed)
    results.append(res)
    if res['success']:
        succ += 1
        times.append(res['time'])
    if (i+1) % 5 == 0:
        print(f"  进度: {i+1}/{REPEAT}")

sr = succ / REPEAT * 100
avg_time = np.mean(times) if times else 0
print(f"\nHCBS-base 成功率: {succ}/{REPEAT} = {sr:.1f}%")
print(f"平均求解时间: {avg_time:.3f}s")

csv_file = os.path.join(OUT_DIR, "hcbs_fixed_results.csv")
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['seed','success','time','makespan','nodes']
    # 如果有error字段则加上
    if results and 'error' in results[0]:
        fieldnames.append('error')
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)
print(f"结果已保存至 {csv_file}")