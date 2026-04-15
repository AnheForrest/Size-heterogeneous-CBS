"""
ablation_auto.py
一键运行四个消融版本：Full, w/o LowSoft, w/o HighHeu, w/o Priority
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
    from reservation_table import ReservationTable
    from sh_agent import AgentClass, AgentInstance
    from congestion_coefficient import compute_cr
    from priority import sort_tasks_by_priority
    import astar
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

# ==================== 固定配置 ====================
TIME_LIMIT = 5.0
MAX_CBS_NODES = 3000
MAX_ATTEMPTS = 1500
REPEAT = 30
MAP_W, MAP_H = 20, 20
OBSTACLE = 0.2
N = 6

SCENE = {
    'name': '混合长条',
    'configs': [
        {'w': 1, 'h': 1, 'ratio': 0.6},
        {'w': 1, 'h': 2, 'ratio': 0.2},
        {'w': 2, 'h': 1, 'ratio': 0.2}
    ]
}

VERSIONS = [
    {'name': 'Full', 'low_soft': True, 'high_heu': True, 'priority': True},
    {'name': 'w_o_LowSoft', 'low_soft': False, 'high_heu': True, 'priority': True},
    {'name': 'w_o_HighHeu', 'low_soft': True, 'high_heu': False, 'priority': True},
    {'name': 'w_o_Priority', 'low_soft': True, 'high_heu': True, 'priority': False},
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUT_DIR = os.path.abspath(os.path.join("..", "test", f"ablation_{timestamp}"))
os.makedirs(BASE_OUT_DIR, exist_ok=True)  # 确保目录存在

# 保存原始权重（若 astar 模块中存在）
ORIG_CR = getattr(astar, 'weight_cr', 0.1)
ORIG_BRIDGE = getattr(astar, 'weight_bridge', 0.3)
ORIG_RES = getattr(astar, 'weight_res', 0.4)

def set_low_soft_weights(enable: bool):
    if enable:
        astar.weight_cr = ORIG_CR
        astar.weight_bridge = ORIG_BRIDGE
        astar.weight_res = ORIG_RES
    else:
        astar.weight_cr = 0.0
        astar.weight_bridge = 0.0
        astar.weight_res = 0.0

def configure_cbs_high_heu(solver: CBS, enable: bool):
    solver.use_severity_sort = enable
    solver.use_dual_constraint = enable
    solver.use_trdp = enable

def run_trial(seed, trial_id, version_config):
    random.seed(seed)
    np.random.seed(seed)
    start = time.time()
    try:
        g_map = GridMap(MAP_W, MAP_H)
        g_map.create_random_obstacles(OBSTACLE)

        agent_classes, counts = [], []
        curr = 0
        for cat_id, cfg in enumerate(SCENE['configs']):
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
            if not pg.V:
                raise ValueError(f"Category {cls.category} no valid positions")
            passable_dict[cls.category] = pg

        agents, _ = generate_tasks(
            agent_classes, counts, g_map, passable_dict,
            existing_occupied=None, max_attempts_per_agent=MAX_ATTEMPTS
        )
        if len(agents) < int(N * 0.8):
            raise Exception(f"Task gen failed: {len(agents)} agents")

        if version_config['priority']:
            agents = sort_tasks_by_priority(agents)

        all_bridges = set()
        for pg in passable_dict.values():
            pg.find_bridges(agent_classes[0])
            all_bridges.update(pg.bridges)
        res_tab = ReservationTable(bridge_cells=all_bridges)
        cr = compute_cr(g_map, passable_dict)

        set_low_soft_weights(version_config['low_soft'])

        solver = CBS(agents, passable_dict, res_tab, cr)
        solver.time_limit = TIME_LIMIT
        configure_cbs_high_heu(solver, version_config['high_heu'])

        success, paths, stats = solver.search(interactive=False)

        elapsed = time.time() - start
        nodes = stats.get('nodes_expanded', 0) if stats else 0
        if elapsed > TIME_LIMIT or nodes > MAX_CBS_NODES:
            success = False
            if stats: stats['reason'] = 'timeout' if elapsed > TIME_LIMIT else 'node_limit'

        return {
            'trial': trial_id, 'seed': seed, 'success': success, 'time': elapsed,
            'makespan': stats.get('makespan', -1) if stats and success else -1,
            'nodes': nodes, 'error': stats.get('reason') if not success else None
        }
    except Exception as e:
        return {'trial': trial_id, 'seed': seed, 'success': False, 'time': time.time()-start,
                'makespan': -1, 'nodes': 0, 'error': str(e)}
    finally:
        set_low_soft_weights(True)

def run_version(version_config):
    name = version_config['name']
    print(f"\n{'='*50}")
    print(f"运行版本: {name}")
    print(f"低层软约束: {version_config['low_soft']}, 高层启发式: {version_config['high_heu']}, 优先级: {version_config['priority']}")
    print(f"{'='*50}")

    results = []
    succ = 0
    times, nodes = [], []
    for i in range(REPEAT):
        seed = hash(f"{name}_{i}") % 100000 + i
        res = run_trial(seed, i+1, version_config)
        results.append(res)
        if res['success']:
            succ += 1
            times.append(res['time'])
            nodes.append(res['nodes'])
        if (i+1) % 10 == 0:
            print(f"  进度: {i+1}/{REPEAT}")

    sr = succ / REPEAT * 100
    avg_time = np.mean(times) if times else 0
    avg_nodes = np.mean(nodes) if nodes else 0
    print(f"成功率: {succ}/{REPEAT} = {sr:.1f}%")
    print(f"平均求解时间: {avg_time:.3f}s")
    print(f"平均扩展节点数: {avg_nodes:.1f}")

    csv_file = os.path.join(BASE_OUT_DIR, f"{name}.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['trial','seed','success','time','makespan','nodes','error'])
        writer.writeheader()
        writer.writerows(results)

    return {'version': name, 'sr': sr, 'avg_time': avg_time, 'avg_nodes': avg_nodes}

def main():
    print(f"开始自动化消融实验，共{len(VERSIONS)}个版本，每版本{REPEAT}次")
    summary = []
    for ver in VERSIONS:
        stats = run_version(ver)
        summary.append(stats)

    print("\n" + "="*60)
    print("消融实验结果汇总")
    print("="*60)
    print(f"{'版本':<20} {'成功率':<10} {'平均时间(s)':<12} {'平均节点数':<10}")
    for s in summary:
        print(f"{s['version']:<20} {s['sr']:.1f}%{'':<6} {s['avg_time']:.3f}{'':<6} {s['avg_nodes']:.1f}")

    summary_csv = os.path.join(BASE_OUT_DIR, "summary.csv")
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['version', 'sr', 'avg_time', 'avg_nodes'])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n汇总结果保存至: {summary_csv}")

if __name__ == "__main__":
    main()