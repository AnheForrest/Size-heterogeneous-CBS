"""
baseline_hard_compare.py
困难场景对比实验：HCBS-base vs 本文算法
场景：长条占比 80%，N=5，20×20，障碍率 0.2
"""

import os
import sys
import time
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def setup_chinese_font():
    font_candidates = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS',
                       'Heiti TC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    for font in font_candidates:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[字体配置] 成功启用中文字体: {font}")
            return
        except:
            continue
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("[警告] 未找到可用中文字体，图表中文可能无法显示。")

setup_chinese_font()

try:
    from gridmap import GridMap
    from passable_graph import PassableGraph
    from task_generator import generate_tasks
    from cbs import CBS
    from cbs_hcbs_base import HCBSBase
    from reservation_table import ReservationTable
    from sh_agent import AgentClass, AgentInstance
    from congestion_coefficient import compute_cr
except ImportError as e:
    print(f"[致命错误] 模块导入失败: {e}")
    sys.exit(1)

# ==================== 实验配置 ====================
TIME_LIMIT = 5.0
MAX_CBS_NODES = 3000
MAX_ATTEMPTS_PER_AGENT = 1500
REPEAT_TIMES = 50
MAP_WIDTH, MAP_HEIGHT = 20, 20
OBSTACLE_RATIO = 0.2

# 困难场景：长条占比 80%，N=5
SCENE_CONFIG = {
    'name': '高长条占比(80%)',
    'configs': [
        {'w': 1, 'h': 1, 'ratio': 0.2},
        {'w': 1, 'h': 2, 'ratio': 0.4},
        {'w': 2, 'h': 1, 'ratio': 0.4}
    ]
}
N_LIST = [5]
ALGORITHMS = ['hcbs_base', 'ours']

# 输出目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR = os.path.join("..", "test", f"baseline_hard_compare_{timestamp}")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
CSV_FILE = os.path.join(BASE_OUTPUT_DIR, "hard_compare_results.csv")
SUMMARY_PNG = os.path.join(BASE_OUTPUT_DIR, "hard_compare.png")

def log_message(msg, to_console=True):
    print(msg)

def run_single_trial(seed, total_agents, trial_id, algorithm='ours'):
    random.seed(seed)
    np.random.seed(seed)
    start_time = time.time()

    try:
        g_map = GridMap(MAP_WIDTH, MAP_HEIGHT)
        g_map.create_random_obstacles(OBSTACLE_RATIO)

        agent_classes = []
        counts = []
        cat_id = 0
        curr_total = 0
        for cfg in SCENE_CONFIG['configs']:
            cnt = int(total_agents * cfg['ratio'])
            cls = AgentClass(category=cat_id, width=cfg['w'], height=cfg['h'])
            agent_classes.append(cls)
            counts.append(cnt)
            curr_total += cnt
            cat_id += 1
        if curr_total < total_agents:
            counts[-1] += (total_agents - curr_total)

        passable_graphs_dict = {}
        for cls in agent_classes:
            pg = PassableGraph(category=cls.category)
            pg.build_from(grid_map=g_map, agent_class=cls)
            pg.width = cls.width
            pg.height = cls.height
            if not pg.V:
                raise ValueError(f"Category {cls.category} has no valid positions")
            passable_graphs_dict[cls.category] = pg

        agents, _ = generate_tasks(
            agent_classes=agent_classes, counts=counts, grid_map=g_map,
            passable_graphs=passable_graphs_dict, existing_occupied=None,
            max_attempts_per_agent=MAX_ATTEMPTS_PER_AGENT
        )
        if len(agents) < int(total_agents * 0.8):
            raise Exception(f"Task generation failed, only {len(agents)} agents")

        if algorithm == 'hcbs_base':
            all_bridges = set()
            for pg in passable_graphs_dict.values():
                pg.find_bridges(agent_classes[0])
                all_bridges.update(pg.bridges)
            res_table = ReservationTable(bridge_cells=all_bridges)
            cr = compute_cr(g_map, passable_graphs_dict)
            solver = HCBSBase(agents, passable_graphs_dict, res_table, cr)
            solver.time_limit = TIME_LIMIT
            success, paths, stats = solver.search()
        else:  # ours
            all_bridges = set()
            for pg in passable_graphs_dict.values():
                pg.find_bridges(agent_classes[0])
                all_bridges.update(pg.bridges)
            res_table = ReservationTable(bridge_cells=all_bridges)
            cr = compute_cr(g_map, passable_graphs_dict)
            solver = CBS(agents, passable_graphs_dict, res_table, cr)
            solver.time_limit = TIME_LIMIT
            success, paths, stats = solver.search(interactive=False)

        elapsed = time.time() - start_time
        nodes = stats.get('nodes_expanded', 0) if stats else 0

        if elapsed > TIME_LIMIT or nodes > MAX_CBS_NODES:
            success = False
            if stats:
                stats['reason'] = 'timeout' if elapsed > TIME_LIMIT else 'node_limit'

        return {
            'algorithm': algorithm,
            'total_agents': total_agents,
            'trial_id': trial_id,
            'seed': seed,
            'success': success,
            'time': elapsed,
            'makespan': stats.get('makespan', -1) if stats and success else -1,
            'cost': stats.get('cost', -1) if stats and success else -1,
            'nodes': nodes,
            'error': stats.get('reason') if not success else None
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'algorithm': algorithm,
            'total_agents': total_agents,
            'trial_id': trial_id,
            'seed': seed,
            'success': False,
            'time': elapsed,
            'makespan': -1,
            'cost': -1,
            'nodes': 0,
            'error': str(e)
        }

def main():
    log_message("="*80)
    log_message("🚀 启动：困难场景对比实验（HCBS-base vs 本文算法）")
    log_message(f"场景: {SCENE_CONFIG['name']}, N = {N_LIST}, 重复: {REPEAT_TIMES}")
    log_message("="*80)

    all_results = []
    total_start = time.time()

    for N in N_LIST:
        log_message(f"\n>>> 智能体数量 N = {N}")
        for alg in ALGORITHMS:
            log_message(f"  运行算法: {alg}")
            succ_count = 0
            times = []
            makespans = []
            for trial in range(REPEAT_TIMES):
                seed = hash(f"hard_{N}_{alg}_{trial}") % 100000 + trial
                res = run_single_trial(seed, N, trial+1, algorithm=alg)
                all_results.append(res)
                if res['success']:
                    succ_count += 1
                    times.append(res['time'])
                    makespans.append(res['makespan'])
                if (trial+1) % 10 == 0:
                    print(f"    进度: {trial+1}/{REPEAT_TIMES}...", end='\r')
            sr = succ_count / REPEAT_TIMES * 100
            avg_time = np.mean(times) if times else 0
            avg_makespan = np.mean(makespans) if makespans else 0
            print(f"    {alg}: SR={sr:.1f}%, Time={avg_time:.3f}s, Makespan={avg_makespan:.1f}")

    total_elapsed = time.time() - total_start
    log_message(f"\n🎉 实验完成，总耗时: {total_elapsed/60:.2f} 分钟")

    if all_results:
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['algorithm', 'total_agents', 'trial_id', 'seed', 'success',
                          'time', 'makespan', 'cost', 'nodes', 'error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        log_message(f"💾 详细数据已保存至: {CSV_FILE}")

        plot_comparison(all_results)
    else:
        log_message("⚠️ 无数据，无法绘图。")

def plot_comparison(results):
    N_values = sorted(list(set(r['total_agents'] for r in results)))
    metrics = ['sr', 'time', 'makespan']
    titles = ['求解成功率 (%)', '平均求解时间 (秒)', '全局完成时间 (Makespan)']
    colors = {'hcbs_base': 'darkorange', 'ours': 'steelblue'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('困难场景下 HCBS-base 与本文算法性能对比', fontsize=36, fontweight='bold')

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(N_values))
        width = 0.35

        for j, alg in enumerate(['hcbs_base', 'ours']):
            vals = []
            for N in N_values:
                sub = [r for r in results if r['total_agents'] == N and r['algorithm'] == alg]
                succ = [r for r in sub if r['success']]
                if metric == 'sr':
                    val = len(succ) / len(sub) * 100 if sub else 0
                elif metric == 'time':
                    val = np.mean([r['time'] for r in succ]) if succ else 0
                else:
                    val = np.mean([r['makespan'] for r in succ]) if succ else 0
                vals.append(val)
            ax.bar(x + (j-0.5)*width, vals, width, label=alg, color=colors[alg], alpha=0.8)

        ax.set_ylabel(titles[i], fontsize=30)
        ax.set_xlabel('智能体数量', fontsize=30)
        ax.set_xticks(x)
        ax.set_xticklabels(N_values, fontsize=30)
        ax.tick_params(axis='y', labelsize=30)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(fontsize=20, loc='upper right')
        if metric == 'sr':
            ax.set_ylim(0, 105)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(SUMMARY_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log_message(f"📈 对比图表已保存: {SUMMARY_PNG}")

if __name__ == "__main__":
    main()