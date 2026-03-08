import os
import sys
import time
import random
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib

# ==========================================================
# 🛠️ 全局配置：中文字体自动适配
# ==========================================================
def setup_chinese_font():
    font_candidates = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Heiti TC', 'WenQuanYi Micro Hei']
    selected = None
    for font in font_candidates:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            selected = font
            break
        except: continue
    if not selected:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        print(f"[系统] 已启用中文字体：{selected}")

setup_chinese_font()

# ==========================================================
# 📦 模块导入
# ==========================================================
try:
    from gridmap import GridMap
    from passable_graph import PassableGraph
    from task_generator import generate_tasks 
    from cbs import CBS
    from reservation_table import ReservationTable
    from sh_agent import AgentClass, AgentInstance
    from visualization import draw_map, draw_agents, draw_path
except ImportError as e:
    print(f"[致命错误] 缺少依赖模块: {e}")
    sys.exit(1)

# ==========================================================
# ⚙️ 【新版】实验参数配置
# ==========================================================
OUTPUT_DIR = "batch_analysis_v2_optimized"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
IMG_DIR = os.path.join(OUTPUT_DIR, "images", "paths")
CSV_FILE = os.path.join(OUTPUT_DIR, "results_v2.csv")
SUMMARY_IMG = os.path.join(OUTPUT_DIR, "performance_comparison_v2.png")

# 地图设置
MAP_WIDTH = 20
MAP_HEIGHT = 20
OBSTACLE_RATIO = 0.1  # 【关键修改】降低障碍率至 10%

# 场景配置 (比例不变)
EXPERIMENT_SCENARIOS = [
    {
        "name": "纯小车 (1x1)",
        "configs": [{'w': 1, 'h': 1, 'ratio': 1.0}]
    },
    {
        "name": "混合方块 (1x1+2x2)",
        "configs": [
            {'w': 1, 'h': 1, 'ratio': 0.8},
            {'w': 2, 'h': 2, 'ratio': 0.2}
        ]
    },
    {
        "name": "混合长条 (1x1+1x2+2x1)",
        "configs": [
            {'w': 1, 'h': 1, 'ratio': 0.6},
            {'w': 1, 'h': 2, 'ratio': 0.2},
            {'w': 2, 'h': 1, 'ratio': 0.2}
        ]
    },
    {
        "name": "复杂异构 (1x1+2x2+3x1)",
        "configs": [
            {'w': 1, 'h': 1, 'ratio': 0.5},
            {'w': 2, 'h': 2, 'ratio': 0.25},
            {'w': 3, 'h': 1, 'ratio': 0.25}
        ]
    }
]

# 【关键修改】只测 6-10，步长 1
AGENT_COUNTS = [6, 7, 8, 9, 10]
REPEAT_TIMES = 20

# 【关键修改】时间限制增加到 10s
TIME_LIMIT_PER_RUN = 10.0
MAX_CBS_NODES = 8000      # 相应增加节点上限
MAX_ATTEMPTS_PER_AGENT = 3000

# 初始化目录
for d in [LOG_DIR, IMG_DIR]:
    os.makedirs(d, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"run_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log(msg, show=True):
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + "\n")
    if show: print(line)

# ==========================================================
# 🧪 核心实验逻辑 (无惩罚时间版)
# ==========================================================

def run_single_trial(seed, n_agents, trial_idx, scenario_name, agent_configs):
    random.seed(seed)
    np.random.seed(seed)
    start_time = time.time()
    
    result = {
        'scenario': scenario_name,
        'n_agents': n_agents,
        'trial': trial_idx,
        'seed': seed,
        'success': False,
        'time_raw': 0.0,       # 实际耗时
        'cost_total': -1,
        'cost_per_agent': -1,
        'makespan': -1,
        'nodes': 0,
        'error_reason': None,
        'img_path': None
    }

    try:
        # 1. 构建地图
        g_map = GridMap(MAP_WIDTH, MAP_HEIGHT)
        g_map.create_random_obstacles(OBSTACLE_RATIO)
        if g_map.grid is None: raise Exception("地图生成失败")

        # 2. 构建智能体
        classes = []
        counts = []
        cat_id = 0
        current_sum = 0
        
        for cfg in agent_configs:
            cnt = int(n_agents * cfg['ratio'])
            cls = AgentClass(category=cat_id, width=cfg['w'], height=cfg['h'])
            classes.append(cls)
            counts.append(cnt)
            current_sum += cnt
            cat_id += 1
        
        if current_sum < n_agents:
            counts[-1] += (n_agents - current_sum)
            
        # 3. 构建可通行图
        p_graphs = {}
        for cls in classes:
            pg = PassableGraph(category=cls.category)
            pg.build_from(grid_map=g_map, agent_class=cls)
            if not pg.V: raise ValueError(f"类别 {cls.category} 无路可走")
            p_graphs[cls.category] = pg

        # 4. 生成任务
        agents, _ = generate_tasks(
            agent_classes=classes, counts=counts, grid_map=g_map,
            passable_graphs=p_graphs, existing_occupied=None,
            max_attempts_per_agent=MAX_ATTEMPTS_PER_AGENT
        )
        
        if len(agents) < int(n_agents * 0.8):
            raise Exception(f"任务生成不足")

        # 5. 运行 CBS
        res_table = ReservationTable(bridge_cells=[])
        cr = [[0]*MAP_WIDTH for _ in range(MAP_HEIGHT)]
        cbs = CBS(agents, p_graphs, res_table, cr)
        cbs.time_limit = TIME_LIMIT_PER_RUN
        
        success, paths, stats = cbs.search(interactive=False)
        
        elapsed = time.time() - start_time
        nodes_expanded = stats.get('nodes_expanded', 0) if stats else 0
        
        # 判定成功与否
        final_success = False
        fail_reason = None
        
        if success:
            if elapsed < TIME_LIMIT_PER_RUN and nodes_expanded <= MAX_CBS_NODES:
                final_success = True
            else:
                fail_reason = "ResourceLimit" # 虽然返回 success 但资源耗尽
        else:
            # 检查失败原因
            if elapsed >= TIME_LIMIT_PER_RUN: fail_reason = "Timeout"
            elif nodes_expanded > MAX_CBS_NODES: fail_reason = "NodeLimit"
            else: fail_reason = "NoSolution"

        result['time_raw'] = elapsed
        result['nodes'] = nodes_expanded
        
        if final_success:
            result['success'] = True
            result['cost_total'] = stats.get('cost', -1)
            result['makespan'] = stats.get('makespan', -1)
            if result['cost_total'] > 0:
                result['cost_per_agent'] = result['cost_total'] / len(agents)
            
            # 保存图片 (每个场景/数量的第 1 次成功)
            if trial_idx == 1:
                safe_name = scenario_name.replace("(", "").replace(")", "").replace(" ", "_")
                img_name = f"{safe_name}_N{n_agents}.png"
                img_path = os.path.join(IMG_DIR, img_name)
                try:
                    fig, ax = plt.subplots(figsize=(6,6))
                    draw_map(g_map, ax=ax, show=False)
                    draw_agents(agents, current_time=None, ax=ax, show=False)
                    for ag in agents:
                        if ag.global_id in paths:
                            draw_path(ag, path=paths[ag.global_id], ax=ax, show=False)
                    ax.set_title(f"{scenario_name}\nN={n_agents}, Time={elapsed:.2f}s")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    result['img_path'] = img_path
                except: pass
        else:
            result['success'] = False
            result['error_reason'] = fail_reason

    except Exception as e:
        elapsed = time.time() - start_time
        result['time_raw'] = elapsed
        result['error_reason'] = str(e)
    
    return result

# ==========================================================
# 📈 数据分析与绘图 (传统指标版)
# ==========================================================

def analyze_and_plot(results):
    scenarios = sorted(list(set(r['scenario'] for r in results)))
    counts = sorted(list(set(r['n_agents'] for r in results)))
    
    agg = {s: {c: {'successes': 0, 'total': 0, 'times': [], 'costs': []} for c in counts} for s in scenarios}
    
    for r in results:
        s, c = r['scenario'], r['n_agents']
        agg[s][c]['total'] += 1
        if r['success']:
            agg[s][c]['successes'] += 1
            agg[s][c]['times'].append(r['time_raw']) # 只用真实时间
            if r['cost_per_agent'] > 0:
                agg[s][c]['costs'].append(r['cost_per_agent'])
        # 失败的不计入时间和成本统计

    stats = {s: {c: {} for c in counts} for s in scenarios}
    
    for s in scenarios:
        for c in counts:
            d = agg[s][c]
            sr = (d['successes'] / d['total']) * 100 if d['total'] > 0 else 0
            
            t_mean = np.mean(d['times']) if d['times'] else 0
            t_std = np.std(d['times']) if len(d['times']) > 1 else 0
            
            cost_mean = np.mean(d['costs']) if d['costs'] else 0
            cost_std = np.std(d['costs']) if len(d['costs']) > 1 else 0
            
            stats[s][c] = {
                'sr': sr,
                'time_mean': t_mean, 'time_std': t_std,
                'cost_mean': cost_mean, 'cost_std': cost_std,
                'total': d['total'], 'successes': d['successes']
            }

    # ================= 绘图 =================
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'CBS 性能评估 (优化版: 障碍率 10%, 时限 10s)', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    
    plots_config = [
        {'key': 'sr', 'title': '求解成功率 (%)', 'ylabel': '成功率 (%)', 'ylim': (0, 105)},
        {'key': 'time', 'title': '平均求解时间 (仅成功案例)', 'ylabel': '时间 (秒)', 'ylim': (0, None)},
        {'key': 'cost', 'title': '单智能体平均路径成本', 'ylabel': '平均成本 (步/车)', 'ylim': (0, None)}
    ]
    
    for i, cfg in enumerate(plots_config):
        ax = axs[i]
        key = cfg['key']
        
        for idx, s in enumerate(scenarios):
            y_means = []
            y_stds = []
            
            for c in counts:
                val = stats[s][c]
                if key == 'sr':
                    y_means.append(val['sr'])
                    y_stds.append(0)
                elif key == 'time':
                    y_means.append(val['time_mean'])
                    y_stds.append(val['time_std'])
                elif key == 'cost':
                    y_means.append(val['cost_mean'])
                    y_stds.append(val['cost_std'])
            
            if key != 'sr':
                ax.errorbar(counts, y_means, yerr=y_stds, fmt='o-', label=s, color=colors[idx],
                            capsize=5, linewidth=2, markersize=6, alpha=0.9)
            else:
                ax.plot(counts, y_means, 'o-', label=s, color=colors[idx], linewidth=2, markersize=6, alpha=0.9)
        
        ax.set_title(cfg['title'], fontsize=12)
        ax.set_xlabel('智能体数量', fontsize=11)
        ax.set_ylabel(cfg['ylabel'], fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', fontsize=9)
        if cfg['ylim'][1] is not None: ax.set_ylim(cfg['ylim'])
        if key == 'sr': ax.set_yticks(range(0, 101, 10))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(SUMMARY_IMG, dpi=300, bbox_inches='tight')
    print(f"\n✅ 优化版性能图已保存：{SUMMARY_IMG}")
    plt.show()

# ==========================================================
# 🚀 主程序入口
# ==========================================================

def main():
    log("="*60)
    log("🚀 启动：CBS 性能优化评测 (v2)")
    log(f"条件：障碍率 10% | 时限 10s | 规模 6-10")
    log("="*60)
    
    all_results = []
    total_start = time.time()
    
    try:
        for scen in EXPERIMENT_SCENARIOS:
            s_name = scen['name']
            s_configs = scen['configs']
            log(f"\n>>> 场景：[{s_name}]")
            
            for n in AGENT_COUNTS:
                batch_start = time.time()
                succ_count = 0
                print(f"  处理 {s_name} | N={n} ...", end='\r')
                
                for i in range(REPEAT_TIMES):
                    seed = hash(f"{s_name}_{n}_{i}_v2") % 100000 + i
                    res = run_single_trial(seed, n, i+1, s_name, s_configs)
                    all_results.append(res)
                    if res['success']: succ_count += 1
                
                duration = time.time() - batch_start
                sr = succ_count / REPEAT_TIMES * 100
                log(f"  [{s_name}] N={n}: 成功率 {sr:.1f}% | 耗时 {duration:.2f}s")
                
        total_dur = time.time() - total_start
        log(f"\n🎉 完成！总耗时：{total_dur/60:.2f} 分钟")
        
        if all_results:
            with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                fields = ['scenario', 'n_agents', 'trial', 'seed', 'success', 'time_raw', 'cost_total', 'cost_per_agent', 'makespan', 'nodes', 'error_reason', 'img_path']
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(all_results)
            log(f"📄 数据已保存：{CSV_FILE}")
            analyze_and_plot(all_results)
            
    except KeyboardInterrupt:
        log("\n⚠️ 中断。保存已有数据...")
        if all_results:
            with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                fields = ['scenario', 'n_agents', 'trial', 'seed', 'success', 'time_raw', 'cost_total', 'cost_per_agent', 'makespan', 'nodes', 'error_reason', 'img_path']
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(all_results)
            analyze_and_plot(all_results)

if __name__ == "__main__":
    main()