"""
ablation_final.py
完整消融实验：5个版本，各50次重复
场景：混合长条 N=6，20×20，障碍率0.2
"""

import os
import sys
import time
import random
import csv
import numpy as np
from datetime import datetime

try:
    from gridmap import GridMap
    from passable_graph import PassableGraph
    from task_generator import generate_tasks
    from cbs import CBS
    from cbs_hcbs_base import HCBSBase
    from reservation_table import ReservationTable
    from sh_agent import AgentClass
    from congestion_coefficient import compute_cr
    from priority import sort_tasks_by_priority
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

TIME_LIMIT = 5.0
MAX_NODES = 3000
REPEAT = 50
MAP_W, MAP_H = 20, 20
OBSTACLE = 0.2
N = 6

SCENE = [
    {'w': 1, 'h': 1, 'ratio': 0.6},
    {'w': 1, 'h': 2, 'ratio': 0.2},
    {'w': 2, 'h': 1, 'ratio': 0.2}
]

# 版本定义
VERSIONS = [
    {'name': 'Full', 'low_soft': True, 'high_heu': True, 'priority': True, 'all_opt': True},
    {'name': 'w/o LowSoft', 'low_soft': False, 'high_heu': True, 'priority': True, 'all_opt': False},
    {'name': 'w/o HighHeu', 'low_soft': True, 'high_heu': False, 'priority': True, 'all_opt': False},
    {'name': 'w/o Priority', 'low_soft': True, 'high_heu': True, 'priority': False, 'all_opt': False},
    {'name': 'w/o AllOpt', 'low_soft': False, 'high_heu': False, 'priority': False, 'all_opt': False},
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("..", "test", f"ablation_final_{timestamp}")
os.makedirs(OUT_DIR, exist_ok=True)

def configure_cbs(solver, version):
    """根据版本配置CBS求解器的优化开关"""
    if not version['all_opt']:
        if version['name'] == 'w/o AllOpt':
            # 直接使用HCBSBase（已关闭所有优化）
            pass
        else:
            # 部分关闭，通过修改solver属性
            if hasattr(solver, 'use_severity_sort'):
                solver.use_severity_sort = version['high_heu']
                solver.use_dual_constraint = version['high_heu']
                solver.use_trdp = version['high_heu']
    # 低层软约束和优先级在外部处理

def run_trial(seed, version):
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

        # 优先级排序
        if version['priority']:
            agents = sort_tasks_by_priority(agents)

        all_bridges = set()
        for pg in passable_dict.values():
            pg.find_bridges(agent_classes[0])
            all_bridges.update(pg.bridges)
        res_tab = ReservationTable(bridge_cells=all_bridges)
        cr = compute_cr(g_map, passable_dict)

        # 设置低层软约束权重
        if version['low_soft']:
            weight_cr, weight_bridge, weight_res = 0.1, 0.3, 0.4
        else:
            weight_cr = weight_bridge = weight_res = 0.0

        if version['name'] == 'w/o AllOpt':
            solver = HCBSBase(agents, passable_dict, res_tab, cr)
        else:
            solver = CBS(agents, passable_dict, res_tab, cr)
            # 注入权重到CBS的_replan方法（通过修改astar模块的全局变量）
            import astar
            astar.weight_cr = weight_cr
            astar.weight_bridge = weight_bridge
            astar.weight_res = weight_res
            configure_cbs(solver, version)

        solver.time_limit = TIME_LIMIT
        success, paths, stats = solver.search(interactive=False)

        elapsed = time.time() - start
        nodes = stats.get('nodes_expanded', 0) if stats else 0
        if elapsed > TIME_LIMIT or nodes > MAX_NODES:
            success = False
            if stats:
                stats['reason'] = 'timeout' if elapsed > TIME_LIMIT else 'node_limit'

        return {
            'version': version['name'], 'seed': seed, 'success': success,
            'time': elapsed, 'makespan': stats.get('makespan', -1) if stats and success else -1,
            'nodes': nodes, 'error': None
        }
    except Exception as e:
        return {'version': version['name'], 'seed': seed, 'success': False,
                'time': time.time()-start, 'makespan': -1, 'nodes': 0, 'error': str(e)}

print(f"开始消融实验，共{len(VERSIONS)}个版本，每版本{REPEAT}次重复")
all_results = []
summary = []

for ver in VERSIONS:
    print(f"\n运行版本: {ver['name']}")
    succ = 0
    times, nodes, makespans = [], [], []
    for i in range(REPEAT):
        seed = hash(f"{ver['name']}_{i}") % 100000 + i
        res = run_trial(seed, ver)
        all_results.append(res)
        if res['success']:
            succ += 1
            times.append(res['time'])
            nodes.append(res['nodes'])
            makespans.append(res['makespan'])
        if (i+1) % 10 == 0:
            print(f"  进度: {i+1}/{REPEAT}")
    sr = succ / REPEAT * 100
    avg_time = np.mean(times) if times else 0
    avg_nodes = np.mean(nodes) if nodes else 0
    avg_makespan = np.mean(makespans) if makespans else 0
    summary.append({
        'version': ver['name'],
        'sr': sr,
        'succ': succ,
        'avg_time': avg_time,
        'avg_nodes': avg_nodes,
        'avg_makespan': avg_makespan
    })
    print(f"  结果: 成功率 {sr:.1f}% ({succ}/{REPEAT}), 平均时间 {avg_time:.3f}s, 平均节点 {avg_nodes:.1f}")

# 保存详细数据
csv_file = os.path.join(OUT_DIR, "ablation_results.csv")
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['version','seed','success','time','makespan','nodes','error']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(all_results)

# 生成汇总表格
print("\n" + "="*70)
print("消融实验结果汇总")
print(f"{'版本':<16} {'成功率':<12} {'平均时间(s)':<14} {'平均节点':<10} {'平均Makespan':<12}")
print("-"*70)
for s in summary:
    print(f"{s['version']:<16} {s['sr']:.1f}% ({s['succ']}/{REPEAT})   {s['avg_time']:.3f}          {s['avg_nodes']:.1f}         {s['avg_makespan']:.1f}")

summary_csv = os.path.join(OUT_DIR, "ablation_summary.csv")
with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['version','sr','succ','avg_time','avg_nodes','avg_makespan'])
    w.writeheader()
    w.writerows(summary)

print(f"\n所有结果保存在: {OUT_DIR}")