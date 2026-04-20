"""
ablation_fine.py
精细消融实验：分温和/困难两种场景，拆解低层软约束子组件
"""

import os, sys, time, random, csv
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
    import astar
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

TIME_LIMIT = 5.0
MAX_NODES = 3000
REPEAT = 50

# ==================== 场景定义 ====================
SCENES = {
    'mild': {
        'name': '温和场景',
        'N': 6,
        'configs': [
            {'w': 1, 'h': 1, 'ratio': 0.6},
            {'w': 1, 'h': 2, 'ratio': 0.2},
            {'w': 2, 'h': 1, 'ratio': 0.2}
        ]
    },
    'hard': {
        'name': '困难场景',
        'N': 5,
        'configs': [
            {'w': 1, 'h': 1, 'ratio': 0.2},
            {'w': 1, 'h': 2, 'ratio': 0.4},
            {'w': 2, 'h': 1, 'ratio': 0.4}
        ]
    }
}

# ==================== 版本定义 ====================
VERSIONS = {
    'Full':      {'cong': True, 'bridge': True, 'res': True, 'high': True, 'prio': True},
    'w/o_Cong':  {'cong': False, 'bridge': True, 'res': True, 'high': True, 'prio': True},
    'w/o_Bridge':{'cong': True, 'bridge': False, 'res': True, 'high': True, 'prio': True},
    'w/o_Res':   {'cong': True, 'bridge': True, 'res': False, 'high': True, 'prio': True},
    'w/o_High':  {'cong': True, 'bridge': True, 'res': True, 'high': False, 'prio': True},
    'w/o_Prio':  {'cong': True, 'bridge': True, 'res': True, 'high': True, 'prio': False},
    'w/o_All':   {'cong': False, 'bridge': False, 'res': False, 'high': False, 'prio': False},
}

# 困难场景仅运行部分版本
HARD_VERSIONS = ['Full', 'w/o_High', 'w/o_Prio', 'w/o_All']

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.abspath(os.path.join("..", "test", f"ablation_fine_{timestamp}"))
os.makedirs(OUT_DIR, exist_ok=True)

def set_astar_weights(cong, bridge, res):
    astar.weight_cr = 0.1 if cong else 0.0
    astar.weight_bridge = 0.3 if bridge else 0.0
    astar.weight_res = 0.4 if res else 0.0

def configure_cbs(solver, high):
    solver.use_severity_sort = high
    solver.use_dual_constraint = high
    solver.use_trdp = high

def run_trial(seed, scene_key, version_key, version_cfg):
    random.seed(seed)
    np.random.seed(seed)
    start = time.time()
    try:
        scene = SCENES[scene_key]
        MAP_W, MAP_H = 20, 20
        OBSTACLE = 0.2
        N = scene['N']

        g_map = GridMap(MAP_W, MAP_H)
        g_map.create_random_obstacles(OBSTACLE)

        agent_classes, counts = [], []
        curr = 0
        for cid, cfg in enumerate(scene['configs']):
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
            raise Exception(f"Task gen failed: {len(agents)}")

        if version_cfg['prio']:
            agents = sort_tasks_by_priority(agents)

        all_bridges = set()
        for pg in passable_dict.values():
            pg.find_bridges(agent_classes[0])
            all_bridges.update(pg.bridges)
        res_tab = ReservationTable(bridge_cells=all_bridges)
        cr = compute_cr(g_map, passable_dict)

        set_astar_weights(version_cfg['cong'], version_cfg['bridge'], version_cfg['res'])

        if version_key == 'w/o_All':
            solver = HCBSBase(agents, passable_dict, res_tab, cr)
        else:
            solver = CBS(agents, passable_dict, res_tab, cr)
            configure_cbs(solver, version_cfg['high'])

        solver.time_limit = TIME_LIMIT
        success, paths, stats = solver.search(interactive=False)

        elapsed = time.time() - start
        nodes = stats.get('nodes_expanded', 0) if stats else 0
        if elapsed > TIME_LIMIT or nodes > MAX_NODES:
            success = False
            if stats: stats['reason'] = 'timeout' if elapsed > TIME_LIMIT else 'node_limit'

        return {
            'scene': scene_key, 'version': version_key, 'seed': seed,
            'success': success, 'time': elapsed,
            'makespan': stats.get('makespan', -1) if stats and success else -1,
            'nodes': nodes, 'error': stats.get('reason') if not success else None
        }
    except Exception as e:
        return {
            'scene': scene_key, 'version': version_key, 'seed': seed,
            'success': False, 'time': time.time()-start,
            'makespan': -1, 'nodes': 0, 'error': str(e)
        }

def run_scene(scene_key, version_list):
    scene_name = SCENES[scene_key]['name']
    print(f"\n{'='*60}")
    print(f"场景: {scene_name}")
    print(f"{'='*60}")
    results = []
    summary = []
    for vkey in version_list:
        vcfg = VERSIONS[vkey]
        print(f"\n版本: {vkey}")
        succ = 0
        times, nodes, makespans = [], [], []
        for i in range(REPEAT):
            seed = hash(f"{scene_key}_{vkey}_{i}") % 100000 + i
            res = run_trial(seed, scene_key, vkey, vcfg)
            results.append(res)
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
            'scene': scene_key, 'version': vkey, 'sr': sr, 'succ': succ,
            'avg_time': avg_time, 'avg_nodes': avg_nodes, 'avg_makespan': avg_makespan
        })
        print(f"  结果: SR={sr:.1f}%, Time={avg_time:.3f}s, Nodes={avg_nodes:.1f}, Makespan={avg_makespan:.1f}")
    return results, summary

def main():
    all_results = []
    all_summary = []
    for scene_key in ['mild', 'hard']:
        version_list = VERSIONS.keys() if scene_key == 'mild' else HARD_VERSIONS
        res, summ = run_scene(scene_key, version_list)
        all_results.extend(res)
        all_summary.extend(summ)

    # 保存详细数据
    csv_file = os.path.join(OUT_DIR, "ablation_fine_results.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['scene','version','seed','success','time','makespan','nodes','error']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)

    # 分场景汇总表格
    for scene_key in ['mild', 'hard']:
        scene_name = SCENES[scene_key]['name']
        print(f"\n{'='*70}")
        print(f"{scene_name} 消融结果汇总")
        print(f"{'版本':<16} {'成功率':<12} {'平均时间(s)':<14} {'平均节点':<10} {'平均Makespan':<12}")
        print("-"*70)
        scene_summary = [s for s in all_summary if s['scene'] == scene_key]
        for s in scene_summary:
            print(f"{s['version']:<16} {s['sr']:.1f}% ({s['succ']}/{REPEAT})   {s['avg_time']:.3f}          {s['avg_nodes']:.1f}         {s['avg_makespan']:.1f}")

    summary_csv = os.path.join(OUT_DIR, "ablation_fine_summary.csv")
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['scene','version','sr','succ','avg_time','avg_nodes','avg_makespan'])
        w.writeheader()
        w.writerows(all_summary)

    print(f"\n所有结果保存在: {OUT_DIR}")

if __name__ == "__main__":
    main()