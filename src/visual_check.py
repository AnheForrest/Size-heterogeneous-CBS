"""
visual_check.py
人工复核 HCBS-base 路径的物理可行性
利用现有 visualization 模块生成逐帧动画和冲突检测
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from gridmap import GridMap
from passable_graph import PassableGraph
from task_generator import generate_tasks
from cbs_hcbs_base import HCBSBase
from reservation_table import ReservationTable
from sh_agent import AgentClass
from congestion_coefficient import compute_cr
from visualization import draw_map, draw_agents, animate_solution
from conflict_detection import detect_conflicts

# ==================== 配置 ====================
MAP_W, MAP_H = 20, 20
OBSTACLE = 0.2
N = 6
SCENE_CONFIGS = [
    {'w': 1, 'h': 1, 'ratio': 0.6},
    {'w': 1, 'h': 2, 'ratio': 0.2},
    {'w': 2, 'h': 1, 'ratio': 0.2}
]

OUTPUT_DIR = "manual_check"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_and_visualize(seed):
    random.seed(seed)
    np.random.seed(seed)

    # --- 地图与任务生成（同前） ---
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

    # --- 求解 ---
    solver = HCBSBase(agents, passable_dict, res_tab, cr)
    solver.time_limit = 5.0
    success, paths, stats = solver.search()
    if not success:
        print(f"Seed {seed} 未成功求解，跳过")
        return

    # 将路径赋给智能体对象，以便可视化模块使用
    for agent in agents:
        agent.set_path(paths[agent.global_id])

    # --- 冲突检测（在真实尺寸下） ---
    # 补齐路径至相同长度
    max_len = max(len(p) for p in paths.values())
    extended_paths = {}
    for aid, path in paths.items():
        if len(path) < max_len:
            extended_paths[aid] = path + [path[-1]] * (max_len - len(path))
        else:
            extended_paths[aid] = path[:max_len]

    temp_res = ReservationTable(bridge_cells=all_bridges)
    for aid, path in extended_paths.items():
        agent = next(a for a in agents if a.global_id == aid)
        temp_res.add(aid, path, agent.agent_class)

    conflicts = detect_conflicts(temp_res, agents, paths_override=extended_paths)
    print(f"Seed {seed}: 检测到 {len(conflicts)} 个冲突")
    for c in conflicts[:5]:  # 只打印前5个
        print(f"  {c}")

    # --- 生成动画（逐帧观察） ---
    anim_path = os.path.join(OUTPUT_DIR, f"hcbs_seed_{seed}.gif")
    print(f"正在生成动画: {anim_path}")
    animate_solution(agents, g_map, interval=500, save_path=anim_path)
    print(f"动画已保存，请逐帧查看是否有矩形重叠")

    # --- 若有冲突，额外保存每个冲突时刻的静态帧 ---
    if conflicts:
        conflict_times = set(c['time'] for c in conflicts if 'time' in c)
        fig, ax = plt.subplots(figsize=(10, 10))
        draw_map(g_map, ax, show=False)
        for t in sorted(conflict_times):
            ax.clear()
            draw_map(g_map, ax, show=False)
            draw_agents(agents, current_time=t, ax=ax, show=False)
            ax.set_title(f"Seed {seed} | Time = {t} | CONFLICT DETECTED", fontsize=14, color='red')
            frame_path = os.path.join(OUTPUT_DIR, f"conflict_seed_{seed}_t{t}.png")
            plt.savefig(frame_path, dpi=150)
            print(f"冲突帧已保存: {frame_path}")
        plt.close(fig)

    # --- 同时生成一张包含所有路径的静态总览图 ---
    fig2, ax2 = plt.subplots(figsize=(12, 12))
    draw_map(g_map, ax2, show=False)
    draw_agents(agents, current_time=None, ax=ax2, show=False)  # 绘制起点/终点
    # 绘制路径线（使用你已有的 draw_path 函数）
    from visualization import draw_path
    for agent in agents:
        draw_path(agent, ax=ax2, show=False)
    ax2.set_title(f"HCBS-base 路径总览 (seed={seed})")
    overview_path = os.path.join(OUTPUT_DIR, f"overview_seed_{seed}.png")
    plt.savefig(overview_path, dpi=150)
    plt.close(fig2)
    print(f"总览图已保存: {overview_path}")

if __name__ == "__main__":
    # 测试几个种子，直到找到一个成功且可能有问题的情况
    test_seeds = [42, 123, 456, 789, 1024, 2023, 4096]
    for s in test_seeds:
        print(f"\n===== 测试 Seed = {s} =====")
        run_and_visualize(s)