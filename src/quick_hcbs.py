"""
quick_32x32_hcbs.py
快速验证 HCBS-base 在 32x32 大规模场景下的表现
"""
import os, sys, time, random, csv
import numpy as np
from datetime import datetime

try:
    from gridmap import GridMap
    from passable_graph import PassableGraph
    from task_generator import generate_tasks
    from cbs_hcbs_base import HCBSBase
    from reservation_table import ReservationTable
    from sh_agent import AgentClass, AgentInstance
    from congestion_coefficient import compute_cr
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

TIME_LIMIT = 5.0
MAX_CBS_NODES = 3000
MAX_ATTEMPTS = 1500
REPEAT = 30
MAP_W, MAP_H = 32, 32
OBSTACLE = 0.2

SCENE = {
    'name': '混合长条',
    'configs': [
        {'w': 1, 'h': 1, 'ratio': 0.6},
        {'w': 1, 'h': 2, 'ratio': 0.2},
        {'w': 2, 'h': 1, 'ratio': 0.2}
    ]
}
N = 13   # 与 3.4 节保持一致

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("..", "test", f"quick_32x32_hcbs_{timestamp}")
os.makedirs(OUT_DIR, exist_ok=True)
CSV = os.path.join(OUT_DIR, "results.csv")

def run_trial(seed, trial_id):
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

        all_bridges = set()
        for pg in passable_dict.values():
            pg.find_bridges(agent_classes[0])
            all_bridges.update(pg.bridges)
        res_tab = ReservationTable(bridge_cells=all_bridges)
        cr = compute_cr(g_map, passable_dict)

        solver = HCBSBase(agents, passable_dict, res_tab, cr)
        solver.time_limit = TIME_LIMIT
        success, paths, stats = solver.search()

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

def main():
    print(f"开始 HCBS-base 32x32 N={N} 测试，重复{REPEAT}次")
    results = []
    succ = 0
    for i in range(REPEAT):
        seed = hash(f"32x32_{i}") % 100000 + i
        res = run_trial(seed, i+1)
        results.append(res)
        if res['success']: succ += 1
        print(f"  {i+1}/{REPEAT}: success={res['success']}, time={res['time']:.2f}s")
    sr = succ / REPEAT * 100
    print(f"\n成功率: {succ}/{REPEAT} = {sr:.1f}%")
    with open(CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['trial','seed','success','time','makespan','nodes','error'])
        w.writeheader()
        w.writerows(results)
    print(f"结果保存至 {CSV}")

if __name__ == "__main__":
    main()