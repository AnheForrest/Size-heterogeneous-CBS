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
    """自动检测并设置中文字体，解决乱码问题"""
    font_candidates = [
        'SimHei',             # Windows: 黑体
        'Microsoft YaHei',    # Windows: 微软雅黑
        'Arial Unicode MS',   # Mac
        'Heiti TC',           # Mac
        'WenQuanYi Micro Hei',# Linux
        'Noto Sans CJK SC'    # Linux
    ]
    selected = None
    for font in font_candidates:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False # 修复负号显示
            selected = font
            break
        except: continue
    
    if not selected:
        print("[警告] 未找到中文字体，强制使用 SimHei (若未安装可能仍显示方块)")
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
    print("请确保 gridmap.py, cbs.py, sh_agent.py 等文件在同一目录下。")
    sys.exit(1)

# ==========================================================
# ⚙️ 实验参数配置
# ==========================================================
OUTPUT_DIR = "batch_performance_analysis"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
IMG_DIR = os.path.join(OUTPUT_DIR, "images", "paths")
CSV_FILE = os.path.join(OUTPUT_DIR, "results_raw.csv")
SUMMARY_IMG = os.path.join(OUTPUT_DIR, "core_performance_triad.png")

# 地图设置
MAP_WIDTH = 20
MAP_HEIGHT = 20
OBSTACLE_RATIO = 0.2

# 【核心实验设计】4 种异构场景
EXPERIMENT_SCENARIOS = [
    {
        "name": "纯小车 (1x1)",
        "desc": "基准组，全同质",
        "configs": [{'w': 1, 'h': 1, 'ratio': 1.0}]
    },
    {
        "name": "混合方块 (1x1+2x2)",
        "desc": "80% 小车 + 20% 大方块",
        "configs": [
            {'w': 1, 'h': 1, 'ratio': 0.8},
            {'w': 2, 'h': 2, 'ratio': 0.2}
        ]
    },
    {
        "name": "混合长条 (1x1+1x2+2x1)",
        "desc": "60% 小车 + 20% 竖条 + 20% 横条",
        "configs": [
            {'w': 1, 'h': 1, 'ratio': 0.6},
            {'w': 1, 'h': 2, 'ratio': 0.2},
            {'w': 2, 'h': 1, 'ratio': 0.2}
        ]
    },
    {
        "name": "复杂异构 (1x1+2x2+3x1)",
        "desc": "50% 小车 + 25% 大方块 + 25% 超长车",
        "configs": [
            {'w': 1, 'h': 1, 'ratio': 0.5},
            {'w': 2, 'h': 2, 'ratio': 0.25},
            {'w': 3, 'h': 1, 'ratio': 0.25}
        ]
    }
]

# 规模梯度：4 到 10，步长 1
AGENT_COUNTS = list(range(4, 11))
REPEAT_TIMES = 20  # 每个配置重复 20 次取平均

# 算法限制
TIME_LIMIT_PER_RUN = 5.0   # 单次求解超时限制 (秒)
MAX_CBS_NODES = 4000       # 节点扩展上限
MAX_ATTEMPTS_PER_AGENT = 2000 # 任务生成尝试次数

# 初始化目录
for d in [LOG_DIR, IMG_DIR]:
    os.makedirs(d, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log(msg, show=True):
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + "\n")
    if show: print(line)

# ==========================================================
# 🧪 核心实验逻辑
# ==========================================================

def run_single_trial(seed, n_agents, trial_idx, scenario_name, agent_configs):
    """
    执行单次实验，返回详细指标字典
    """
    random.seed(seed)
    np.random.seed(seed)
    start_time = time.time()
    
    result = {
        'scenario': scenario_name,
        'n_agents': n_agents,
        'trial': trial_idx,
        'seed': seed,
        'success': False,
        'time_raw': 0.0,
        'time_effective': 0.0, # 含惩罚的时间
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
        if g_map.grid is None:
            raise Exception("地图生成失败")

        # 2. 构建智能体类别与数量分配
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
        
        # 补齐余数到最后一种类
        if current_sum < n_agents:
            counts[-1] += (n_agents - current_sum)
            
        # 3. 构建可通行图 (Passable Graph)
        p_graphs = {}
        for cls in classes:
            pg = PassableGraph(category=cls.category)
            pg.build_from(grid_map=g_map, agent_class=cls)
            if not pg.V:
                raise ValueError(f"类别 {cls.category} ({cls.width}x{cls.height}) 无路可走")
            p_graphs[cls.category] = pg

        # 4. 生成任务
        agents, _ = generate_tasks(
            agent_classes=classes, counts=counts, grid_map=g_map,
            passable_graphs=p_graphs, existing_occupied=None,
            max_attempts_per_agent=MAX_ATTEMPTS_PER_AGENT
        )
        
        if len(agents) < int(n_agents * 0.8):
            raise Exception(f"任务生成不足 (仅 {len(agents)}/{n_agents})")

        # 5. 运行 CBS 算法
        res_table = ReservationTable(bridge_cells=[])
        cr = [[0]*MAP_WIDTH for _ in range(MAP_HEIGHT)]
        cbs = CBS(agents, p_graphs, res_table, cr)
        cbs.time_limit = TIME_LIMIT_PER_RUN
        
        success, paths, stats = cbs.search(interactive=False)
        
        elapsed = time.time() - start_time
        nodes_expanded = stats.get('nodes_expanded', 0) if stats else 0
        
        # 判定是否因资源限制而失败
        forced_fail = False
        fail_reason = None
        
        if not success:
            # 检查是否是超时或节点过多导致的“假性无解”
            if elapsed >= TIME_LIMIT_PER_RUN:
                forced_fail = True
                fail_reason = "Timeout"
            elif nodes_expanded > MAX_CBS_NODES:
                forced_fail = True
                fail_reason = "NodeLimit"
            else:
                fail_reason = "NoSolution" # 真的无解
        else:
            # 即使返回 success，也要检查是否刚好踩线
            if elapsed >= TIME_LIMIT_PER_RUN or nodes_expanded > MAX_CBS_NODES:
                # 这种情况极少，但为了严谨，视为有效但高风险
                pass 

        final_success = success and not forced_fail
        
        # === 计算核心指标 ===
        result['time_raw'] = elapsed
        result['nodes'] = nodes_expanded
        
        # 【关键修正 1】有效时间：失败则计入超时惩罚
        if final_success:
            result['success'] = True
            result['time_effective'] = elapsed
            result['cost_total'] = stats.get('cost', -1)
            result['makespan'] = stats.get('makespan', -1)
            result['error_reason'] = None
            
            # 【关键修正 2】单智能体成本
            if result['cost_total'] > 0:
                result['cost_per_agent'] = result['cost_total'] / len(agents)
            
            # 保存首条成功路径图 (仅每个场景/数量的第 1 次成功)
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
                    ax.set_title(f"{scenario_name}\nN={n_agents}, Cost={result['cost_total']}")
                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    result['img_path'] = img_path
                except Exception as e:
                    pass
                    
        else:
            result['success'] = False
            result['time_effective'] = TIME_LIMIT_PER_RUN # 惩罚时间
            result['error_reason'] = fail_reason

    except Exception as e:
        elapsed = time.time() - start_time
        result['time_raw'] = elapsed
        result['time_effective'] = TIME_LIMIT_PER_RUN # 异常也视为超时惩罚
        result['error_reason'] = str(e)
    
    return result

# ==========================================================
# 📈 数据分析与绘图
# ==========================================================

def analyze_and_plot(results):
    """
    分析数据并绘制“性能铁三角”图表
    """
    # 数据结构整理
    scenarios = sorted(list(set(r['scenario'] for r in results)))
    counts = sorted(list(set(r['n_agents'] for r in results)))
    
    # 初始化聚合容器
    # data[scenario][count] = { 'sr': [], 'time_eff': [], 'cost_avg': [] }
    agg = {s: {c: {'successes': 0, 'total': 0, 'times': [], 'costs': []} for c in counts} for s in scenarios}
    
    for r in results:
        s, c = r['scenario'], r['n_agents']
        agg[s][c]['total'] += 1
        if r['success']:
            agg[s][c]['successes'] += 1
            agg[s][c]['times'].append(r['time_effective']) # 成功用实际时间
            if r['cost_per_agent'] > 0:
                agg[s][c]['costs'].append(r['cost_per_agent'])
        else:
            agg[s][c]['times'].append(r['time_effective']) # 失败用惩罚时间 (5s)
            # 失败没有成本数据，不计入成本平均，或者可以计为无穷大（这里选择不计入，以免破坏均值意义，但在论文中需说明）

    # 计算统计量
    stats = {s: {c: {} for c in counts} for s in scenarios}
    
    for s in scenarios:
        for c in counts:
            d = agg[s][c]
            # 1. 成功率
            sr = (d['successes'] / d['total']) * 100 if d['total'] > 0 else 0
            
            # 2. 平均有效时间 (含惩罚)
            t_mean = np.mean(d['times']) if d['times'] else 0
            t_std = np.std(d['times']) if len(d['times']) > 1 else 0
            
            # 3. 平均单智能体成本 (仅基于成功案例)
            if d['costs']:
                cost_mean = np.mean(d['costs'])
                cost_std = np.std(d['costs'])
            else:
                cost_mean = 0 # 无成功数据
                cost_std = 0
            
            stats[s][c] = {
                'sr': sr,
                'time_mean': t_mean, 'time_std': t_std,
                'cost_mean': cost_mean, 'cost_std': cost_std
            }

    # ================= 绘图 =================
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('CBS 算法核心性能三维评估 (速度 - 质量 - 成功率)', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    
    # 配置三个子图的元数据
    plots_config = [
        {
            'key': 'sr',
            'title': '求解成功率 (鲁棒性)',
            'ylabel': '成功率 (%)',
            'ylim': (0, 105),
            'show_std': False
        },
        {
            'key': 'time',
            'title': '期望求解耗时 (含超时惩罚)',
            'ylabel': '时间 (秒)',
            'ylim': (0, None), # 自动适应
            'show_std': True
        },
        {
            'key': 'cost',
            'title': '单智能体平均路径成本 (质量)',
            'ylabel': '平均成本 (步/车)',
            'ylim': (0, None),
            'show_std': True
        }
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
            
            # 绘制误差棒图
            if cfg['show_std']:
                ax.errorbar(counts, y_means, yerr=y_stds, fmt='o-', label=s, color=colors[idx],
                            capsize=5, linewidth=2, markersize=6, alpha=0.9)
            else:
                ax.plot(counts, y_means, 'o-', label=s, color=colors[idx], linewidth=2, markersize=6, alpha=0.9)
        
        ax.set_title(cfg['title'], fontsize=12)
        ax.set_xlabel('智能体数量', fontsize=11)
        ax.set_ylabel(cfg['ylabel'], fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', fontsize=9)
        
        if cfg['ylim'][1] is not None:
            ax.set_ylim(cfg['ylim'])
        if key == 'sr':
            ax.set_yticks(range(0, 101, 20))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(SUMMARY_IMG, dpi=300, bbox_inches='tight')
    print(f"\n✅ 性能分析图已保存：{SUMMARY_IMG}")
    plt.show()

# ==========================================================
# 🚀 主程序入口
# ==========================================================

def main():
    log("="*60)
    log("🚀 启动：异构多智能体路径规划性能深度评测")
    log(f"场景数：{len(EXPERIMENT_SCENARIOS)}")
    log(f"规模范围：{min(AGENT_COUNTS)} - {max(AGENT_COUNTS)}")
    log(f"重复次数：{REPEAT_TIMES}")
    log(f"超时限制：{TIME_LIMIT_PER_RUN}s | 节点限制：{MAX_CBS_NODES}")
    log("="*60)
    
    all_results = []
    total_start = time.time()
    
    try:
        for scen in EXPERIMENT_SCENARIOS:
            s_name = scen['name']
            s_configs = scen['configs']
            log(f"\n>>> 开始场景：[{s_name}]")
            
            for n in AGENT_COUNTS:
                batch_start = time.time()
                succ_count = 0
                
                # 进度提示
                print(f"  处理 {s_name} | N={n} ...", end='\r')
                
                for i in range(REPEAT_TIMES):
                    seed = hash(f"{s_name}_{n}_{i}") % 100000 + i
                    res = run_single_trial(seed, n, i+1, s_name, s_configs)
                    all_results.append(res)
                    if res['success']:
                        succ_count += 1
                
                # 本组小结
                duration = time.time() - batch_start
                sr = succ_count / REPEAT_TIMES * 100
                log(f"  [{s_name}] N={n}: 成功率 {sr:.1f}% | 耗时 {duration:.2f}s")
                
        total_dur = time.time() - total_start
        log(f"\n🎉 全部完成！总耗时：{total_dur/60:.2f} 分钟")
        
        # 保存原始 CSV
        if all_results:
            with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                fields = ['scenario', 'n_agents', 'trial', 'seed', 'success', 'time_raw', 'time_effective', 'cost_total', 'cost_per_agent', 'makespan', 'nodes', 'error_reason', 'img_path']
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(all_results)
            log(f"📄 原始数据已保存：{CSV_FILE}")
            
            # 生成图表
            analyze_and_plot(all_results)
        else:
            log("⚠️ 未收集到任何数据，跳过绘图。")
            
    except KeyboardInterrupt:
        log("\n⚠️ 用户中断实验。将保存已有数据并绘图。")
        if all_results:
            # 保存部分数据
            with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                fields = ['scenario', 'n_agents', 'trial', 'seed', 'success', 'time_raw', 'time_effective', 'cost_total', 'cost_per_agent', 'makespan', 'nodes', 'error_reason', 'img_path']
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(all_results)
            analyze_and_plot(all_results)

if __name__ == "__main__":
    main()