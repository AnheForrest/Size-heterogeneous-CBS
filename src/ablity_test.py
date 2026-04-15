"""
factor_analysis.py
性能影响因素分析实验脚本（方案C-精细化版）
探究障碍率、长条占比、地图尺寸对算法性能的影响
基线：混合长条场景，N=5，重复100次
"""

import os
import sys
import time
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib

# ==================== 中文字体配置 ====================
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

# ==================== 导入核心模块 ====================
try:
    from gridmap import GridMap
    from passable_graph import PassableGraph
    from task_generator import generate_tasks
    from cbs import CBS
    from reservation_table import ReservationTable
    from sh_agent import AgentClass, AgentInstance
    from visualization import draw_map, draw_agents, draw_path
except ImportError as e:
    print(f"[致命错误] 模块导入失败: {e}")
    sys.exit(1)

# ==================== 固定参数（不随因素改变） ====================
TIME_LIMIT_PER_RUN = 5.0
MAX_CBS_NODES = 3000
MAX_ATTEMPTS_PER_AGENT = 1500
REPEAT_TIMES = 100          # 每个配置重复次数

# ==================== 输出目录 ====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR = os.path.join("..", "test", f"factor_analysis_v3_fine_{timestamp}")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

LOG_DIR = os.path.join(BASE_OUTPUT_DIR, "logs")
IMG_DIR = os.path.join(BASE_OUTPUT_DIR, "images")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

log_filename = f"log_factor_{timestamp}.txt"
log_path = os.path.join(LOG_DIR, log_filename)

def log_message(msg, to_console=True):
    """记录日志"""
    time_str = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{time_str}] {msg}"
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(full_msg + "\n")
    if to_console:
        print(full_msg)

# ==================== 核心测试函数（增加障碍率和地图尺寸参数） ====================
def run_single_trial(seed, total_agents, trial_id, scenario_name, agent_configs,
                     map_width=20, map_height=20, obstacle_ratio=0.2):
    """
    单次实验运行
    :param map_width, map_height: 可传入不同地图尺寸
    :param obstacle_ratio: 可传入不同障碍率
    """
    random.seed(seed)
    np.random.seed(seed)
    start_time = time.time()

    try:
        # 使用传入的地图参数创建地图
        g_map = GridMap(map_width, map_height)
        g_map.create_random_obstacles(obstacle_ratio)
        if g_map.grid is None:
            raise Exception("Map creation failed")

        # 构建智能体类别
        agent_classes = []
        counts = []
        cat_id = 0
        curr_total = 0
        for cfg in agent_configs:
            cnt = int(total_agents * cfg['ratio'])
            cls = AgentClass(category=cat_id, width=cfg['w'], height=cfg['h'])
            agent_classes.append(cls)
            counts.append(cnt)
            curr_total += cnt
            cat_id += 1
        if curr_total < total_agents:
            counts[-1] += (total_agents - curr_total)

        # 构建可通行子图
        passable_graphs_dict = {}
        for cls in agent_classes:
            pg = PassableGraph(category=cls.category)
            pg.build_from(grid_map=g_map, agent_class=cls)
            if not pg.V:
                raise ValueError(f"Category {cls.category} has no valid positions")
            passable_graphs_dict[cls.category] = pg

        # 生成任务
        agents, _ = generate_tasks(
            agent_classes=agent_classes, counts=counts, grid_map=g_map,
            passable_graphs=passable_graphs_dict, existing_occupied=None,
            max_attempts_per_agent=MAX_ATTEMPTS_PER_AGENT
        )
        if len(agents) < int(total_agents * 0.8):
            raise Exception(f"Task generation failed, only {len(agents)} agents")

        # 准备 CBS 求解器
        res_table = ReservationTable(bridge_cells=[])
        cr = [[0] * map_width for _ in range(map_height)]
        cbs_solver = CBS(agents, passable_graphs_dict, res_table, cr)
        cbs_solver.time_limit = TIME_LIMIT_PER_RUN

        success, paths, stats = cbs_solver.search(interactive=False)
        elapsed = time.time() - start_time
        nodes = stats.get('nodes_expanded', 0) if stats else 0

        if elapsed > TIME_LIMIT_PER_RUN or nodes > MAX_CBS_NODES:
            success = False
            reason = "超时" if elapsed > TIME_LIMIT_PER_RUN else "节点过多"
            if stats:
                stats['reason'] = reason

        result = {
            'scenario': scenario_name,
            'seed': seed,
            'total_agents': total_agents,
            'trial_id': trial_id,
            'success': success,
            'time': elapsed,
            'makespan': stats.get('makespan', -1) if stats and success else -1,
            'cost': stats.get('cost', -1) if stats and success else -1,
            'nodes': nodes,
            'generated_count': len(agents),
            'error': stats.get('reason', None) if not success else None,
            'map_width': map_width,
            'map_height': map_height,
            'obstacle_ratio': obstacle_ratio
        }
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'scenario': scenario_name, 'seed': seed, 'total_agents': total_agents,
            'trial_id': trial_id, 'success': False, 'time': elapsed, 'error': str(e),
            'makespan': -1, 'cost': -1, 'nodes': 0, 'generated_count': 0,
            'map_width': map_width, 'map_height': map_height, 'obstacle_ratio': obstacle_ratio
        }

# ==================== 辅助函数：运行一组实验并保存结果 ====================
def run_experiment_group(exp_name, varying_param_name, varying_values,
                         fixed_scene_config, fixed_N, fixed_map_size=(20,20),
                         fixed_obstacle_ratio=0.2):
    """
    运行一组单因素实验
    """
    log_message(f"\n{'='*60}")
    log_message(f"开始实验组: {exp_name}")
    log_message(f"变化参数: {varying_param_name} = {varying_values}")
    log_message(f"固定配置: N={fixed_N}, 场景={fixed_scene_config['name']}")
    log_message(f"{'='*60}")

    all_results = []
    summary_stats = {}

    for val in varying_values:
        if varying_param_name == 'obstacle_ratio':
            obs_ratio = val
            map_w, map_h = fixed_map_size
        elif varying_param_name == 'map_size':
            map_w, map_h = val
            obs_ratio = fixed_obstacle_ratio
        else:
            map_w, map_h = fixed_map_size
            obs_ratio = fixed_obstacle_ratio

        log_message(f"\n>>> 当前 {varying_param_name} = {val}")

        succ_count = 0
        group_times = []
        group_makespans = []
        group_costs = []

        for i in range(REPEAT_TIMES):
            seed = hash(f"{exp_name}_{val}_{i}") % 100000 + i
            res = run_single_trial(
                seed=seed,
                total_agents=fixed_N,
                trial_id=i+1,
                scenario_name=fixed_scene_config['name'],
                agent_configs=fixed_scene_config['configs'],
                map_width=map_w,
                map_height=map_h,
                obstacle_ratio=obs_ratio
            )
            res['exp_group'] = exp_name
            res['varying_param'] = varying_param_name
            res['param_value'] = str(val)
            all_results.append(res)

            if res['success']:
                succ_count += 1
                group_times.append(res['time'])
                group_makespans.append(res['makespan'])
                group_costs.append(res['cost'])

            if (i+1) % 10 == 0:
                print(f"  进度: {i+1}/{REPEAT_TIMES}...", end='\r')

        sr = succ_count / REPEAT_TIMES * 100
        avg_time = np.mean(group_times) if group_times else 0
        avg_makespan = np.mean(group_makespans) if group_makespans else 0
        avg_cost = np.mean(group_costs) if group_costs else 0

        print(f"  {varying_param_name}={val}: SR={sr:.1f}%, Time={avg_time:.3f}s, Makespan={avg_makespan:.1f}")
        log_message(f"  {varying_param_name}={val} -> SR: {sr:.1f}%, Time: {avg_time:.3f}s")

        if varying_param_name == 'map_size':
            key = f"{map_w}×{map_h}"
        else:
            key = val
        summary_stats[key] = {
            'sr': sr,
            'time': avg_time,
            'makespan': avg_makespan,
            'cost': avg_cost,
            'success_count': succ_count
        }

    csv_file = os.path.join(BASE_OUTPUT_DIR, f"{exp_name.replace(' ', '_')}.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['exp_group', 'varying_param', 'param_value', 'scenario', 'seed',
                      'total_agents', 'trial_id', 'success', 'time', 'makespan', 'cost',
                      'nodes', 'error', 'generated_count', 'map_width', 'map_height', 'obstacle_ratio']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    log_message(f"💾 详细数据已保存至: {csv_file}")

    plot_single_factor_chart(exp_name, varying_param_name, summary_stats)
    return summary_stats

def plot_single_factor_chart(exp_name, param_name, stats_dict):
    """为单因素实验绘制对比柱状图（字体按论文要求：刻度30，轴标题20，总标题42）"""
    if param_name == 'map_size':
        items = sorted(stats_dict.items(), key=lambda x: int(x[0].split('×')[0]) * int(x[0].split('×')[1]))
    else:
        items = sorted(stats_dict.items(), key=lambda x: x[0])

    labels = [str(k) for k, v in items]
    sr_vals = [v['sr'] for k, v in items]
    time_vals = [v['time'] for k, v in items]
    makespan_vals = [v['makespan'] for k, v in items]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{exp_name} 对算法性能的影响', fontsize=30, fontweight='bold')

    # 成功率子图
    axes[0].bar(labels, sr_vals, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('求解成功率 (%)', fontsize=30)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].tick_params(axis='both', labelsize=20)

    # 求解时间子图
    axes[1].bar(labels, time_vals, color='darkorange', alpha=0.8)
    axes[1].set_ylabel('平均求解时间 (秒)', fontsize=30)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='both', labelsize=20)

    # Makespan子图
    axes[2].bar(labels, makespan_vals, color='seagreen', alpha=0.8)
    axes[2].set_ylabel('全局完成时间 (步)', fontsize=30)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='both', labelsize=20)

    # 设置x轴标签
    if param_name == 'obstacle_ratio':
        xlabel = '障碍率'
    elif param_name == 'map_size':
        xlabel = '地图尺寸'
    elif param_name == 'long_ratio':
        xlabel = '长条智能体占比'
    else:
        xlabel = param_name
    for ax in axes:
        ax.set_xlabel(xlabel, fontsize=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(IMG_DIR, f"{exp_name.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log_message(f"📈 对比图表已保存: {save_path}")

# ==================== 定义三个影响因素实验的配置（精细化版） ====================

# 1. 障碍率影响实验（5档）
EXP1_OBSTACLE = {
    'name': '障碍率影响',
    'varying_param': 'obstacle_ratio',
    'values': [0.10, 0.15, 0.20, 0.25, 0.30],
    'scene': {
        'name': '混合长条 (固定)',
        'configs': [
            {'w': 1, 'h': 1, 'ratio': 0.6},
            {'w': 1, 'h': 2, 'ratio': 0.2},
            {'w': 2, 'h': 1, 'ratio': 0.2}
        ]
    },
    'fixed_N': 5,
    'fixed_map_size': (20, 20),
    'fixed_obstacle_ratio': None
}

# 2. 长条智能体占比影响实验（4档：20%、40%、60%、80%）
EXP2_RATIO = {
    'name': '长条占比影响',
    'varying_param': 'long_ratio',
    'values': ['20%', '40%', '60%', '80%'],
    'scene_configs': {
        '20%': [
            {'w': 1, 'h': 1, 'ratio': 0.8},
            {'w': 1, 'h': 2, 'ratio': 0.1},
            {'w': 2, 'h': 1, 'ratio': 0.1}
        ],
        '40%': [
            {'w': 1, 'h': 1, 'ratio': 0.6},
            {'w': 1, 'h': 2, 'ratio': 0.2},
            {'w': 2, 'h': 1, 'ratio': 0.2}
        ],
        '60%': [
            {'w': 1, 'h': 1, 'ratio': 0.4},
            {'w': 1, 'h': 2, 'ratio': 0.3},
            {'w': 2, 'h': 1, 'ratio': 0.3}
        ],
        '80%': [
            {'w': 1, 'h': 1, 'ratio': 0.2},
            {'w': 1, 'h': 2, 'ratio': 0.4},
            {'w': 2, 'h': 1, 'ratio': 0.4}
        ]
    },
    'fixed_N': 5,
    'fixed_map_size': (20, 20),
    'fixed_obstacle_ratio': 0.2
}

# 3. 地图尺寸影响实验（5档：20→23→26→29→32）
EXP3_MAPSIZE = {
    'name': '地图尺寸影响',
    'varying_param': 'map_size',
    'values': [(20, 20), (23, 23), (26, 26), (29, 29), (32, 32)],
    'scene': {
        'name': '混合长条 (固定)',
        'configs': [
            {'w': 1, 'h': 1, 'ratio': 0.6},
            {'w': 1, 'h': 2, 'ratio': 0.2},
            {'w': 2, 'h': 1, 'ratio': 0.2}
        ]
    },
    'fixed_N_map': {
        (20, 20): 5,
        (23, 23): 7,
        (26, 26): 8,
        (29, 29): 11,
        (32, 32): 13
    },
    'fixed_obstacle_ratio': 0.2
}

# ==================== 主流程 ====================
def main():
    log_message("="*80)
    log_message("🚀 启动：性能影响因素分析实验（精细化版）")
    log_message(f"输出目录: {BASE_OUTPUT_DIR}")
    log_message(f"每组重复次数: {REPEAT_TIMES}")
    log_message("="*80)

    total_start = time.time()

    # ---------- 实验1：障碍率影响 ----------
    exp1 = EXP1_OBSTACLE
    run_experiment_group(
        exp_name=exp1['name'],
        varying_param_name=exp1['varying_param'],
        varying_values=exp1['values'],
        fixed_scene_config=exp1['scene'],
        fixed_N=exp1['fixed_N'],
        fixed_map_size=exp1['fixed_map_size']
    )

    # ---------- 实验2：长条占比影响 ----------
    exp2 = EXP2_RATIO
    log_message(f"\n{'='*60}")
    log_message(f"开始实验组: {exp2['name']}")
    log_message(f"变化参数: 长条(1x2+2x1)占比 = {exp2['values']}")
    log_message(f"固定配置: N={exp2['fixed_N']}, 障碍率={exp2['fixed_obstacle_ratio']}")
    log_message(f"{'='*60}")

    all_results_ratio = []
    summary_ratio = {}

    for ratio_label in exp2['values']:
        configs = exp2['scene_configs'][ratio_label]
        scene_name = f"长条占比{ratio_label}"
        log_message(f"\n>>> 当前长条占比 = {ratio_label}")

        succ_count = 0
        group_times = []
        group_makespans = []

        for i in range(REPEAT_TIMES):
            seed = hash(f"long_ratio_{ratio_label}_{i}") % 100000 + i
            res = run_single_trial(
                seed=seed,
                total_agents=exp2['fixed_N'],
                trial_id=i+1,
                scenario_name=scene_name,
                agent_configs=configs,
                map_width=exp2['fixed_map_size'][0],
                map_height=exp2['fixed_map_size'][1],
                obstacle_ratio=exp2['fixed_obstacle_ratio']
            )
            res['exp_group'] = exp2['name']
            res['varying_param'] = 'long_ratio'
            res['param_value'] = ratio_label
            all_results_ratio.append(res)

            if res['success']:
                succ_count += 1
                group_times.append(res['time'])
                group_makespans.append(res['makespan'])

            if (i+1) % 10 == 0:
                print(f"  进度: {i+1}/{REPEAT_TIMES}...", end='\r')

        sr = succ_count / REPEAT_TIMES * 100
        avg_time = np.mean(group_times) if group_times else 0
        avg_makespan = np.mean(group_makespans) if group_makespans else 0

        print(f"  长条占比={ratio_label}: SR={sr:.1f}%, Time={avg_time:.3f}s, Makespan={avg_makespan:.1f}")
        log_message(f"  长条占比={ratio_label} -> SR: {sr:.1f}%, Time: {avg_time:.3f}s")

        summary_ratio[ratio_label] = {
            'sr': sr, 'time': avg_time, 'makespan': avg_makespan,
            'cost': 0, 'success_count': succ_count
        }

    csv_ratio = os.path.join(BASE_OUTPUT_DIR, "long_ratio_analysis.csv")
    with open(csv_ratio, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['exp_group', 'varying_param', 'param_value', 'scenario', 'seed',
                      'total_agents', 'trial_id', 'success', 'time', 'makespan', 'cost',
                      'nodes', 'error', 'generated_count', 'map_width', 'map_height', 'obstacle_ratio']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results_ratio)
    log_message(f"💾 详细数据已保存至: {csv_ratio}")

    plot_single_factor_chart(exp2['name'], 'long_ratio', summary_ratio)

    # ---------- 实验3：地图尺寸影响 ----------
    exp3 = EXP3_MAPSIZE
    log_message(f"\n{'='*60}")
    log_message(f"开始实验组: {exp3['name']}")
    log_message(f"变化参数: 地图尺寸 = {exp3['values']}")
    log_message(f"固定配置: 障碍率={exp3['fixed_obstacle_ratio']}, 场景=混合长条")
    log_message(f"{'='*60}")

    all_results_map = []
    summary_map = {}

    for map_size in exp3['values']:
        w, h = map_size
        N = exp3['fixed_N_map'][map_size]
        log_message(f"\n>>> 当前地图尺寸 = {w}×{h}, N = {N}")

        succ_count = 0
        group_times = []
        group_makespans = []

        for i in range(REPEAT_TIMES):
            seed = hash(f"map_{w}_{h}_{i}") % 100000 + i
            res = run_single_trial(
                seed=seed,
                total_agents=N,
                trial_id=i+1,
                scenario_name=exp3['scene']['name'],
                agent_configs=exp3['scene']['configs'],
                map_width=w,
                map_height=h,
                obstacle_ratio=exp3['fixed_obstacle_ratio']
            )
            res['exp_group'] = exp3['name']
            res['varying_param'] = 'map_size'
            res['param_value'] = f"{w}x{h}"
            all_results_map.append(res)

            if res['success']:
                succ_count += 1
                group_times.append(res['time'])
                group_makespans.append(res['makespan'])

            if (i+1) % 10 == 0:
                print(f"  进度: {i+1}/{REPEAT_TIMES}...", end='\r')

        sr = succ_count / REPEAT_TIMES * 100
        avg_time = np.mean(group_times) if group_times else 0
        avg_makespan = np.mean(group_makespans) if group_makespans else 0

        print(f"  {w}×{h}: SR={sr:.1f}%, Time={avg_time:.3f}s, Makespan={avg_makespan:.1f}")
        log_message(f"  {w}×{h} -> SR: {sr:.1f}%, Time: {avg_time:.3f}s")

        summary_map[f"{w}×{h}"] = {
            'sr': sr, 'time': avg_time, 'makespan': avg_makespan,
            'cost': 0, 'success_count': succ_count
        }

    csv_map = os.path.join(BASE_OUTPUT_DIR, "map_size_analysis.csv")
    with open(csv_map, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['exp_group', 'varying_param', 'param_value', 'scenario', 'seed',
                      'total_agents', 'trial_id', 'success', 'time', 'makespan', 'cost',
                      'nodes', 'error', 'generated_count', 'map_width', 'map_height', 'obstacle_ratio']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results_map)
    log_message(f"💾 详细数据已保存至: {csv_map}")

    plot_single_factor_chart(exp3['name'], '地图尺寸', summary_map)

    # ---------- 完成 ----------
    total_elapsed = time.time() - total_start
    log_message(f"\n{'='*80}")
    log_message(f"🎉 所有影响因素分析实验完成！")
    log_message(f"总耗时: {total_elapsed/60:.2f} 分钟")
    log_message(f"所有结果已保存至: {BASE_OUTPUT_DIR}")
    log_message(f"{'='*80}")

if __name__ == "__main__":
    main()