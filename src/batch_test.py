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

def setup_chinese_font():
    """
    自动检测并设置中文字体，解决图表中文显示为方块或空白的问题
    """
    # 常见中文字体列表，按优先级排序
    font_candidates = [
        'SimHei',             # Windows: 黑体
        'Microsoft YaHei',    # Windows: 微软雅黑
        'Arial Unicode MS',   # Mac: 常用
        'Heiti TC',           # Mac: 黑体
        'WenQuanYi Micro Hei',# Linux: 文泉驿
        'Noto Sans CJK SC'    # Linux/Google: 思源黑体
    ]
    
    selected_font = None
    for font in font_candidates:
        try:
            # 尝试设置，如果不报错说明系统有这个字体
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
            selected_font = font
            break
        except Exception:
            continue
    
    if selected_font:
        print(f"[字体配置] 成功启用中文字体: {selected_font}")
    else:
        print("[警告] 未找到可用中文字体，图表中文可能无法显示。建议安装 'SimHei' 或 'Microsoft YaHei'。")
        #  fallback: 即使没找到，也强行设置一个常见的，让用户手动安装
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False

# 在脚本启动时立即执行字体配置
setup_chinese_font()

# ==========================================================
# 导入其他模块
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
    print(f"[致命错误] 模块导入失败: {e}")
    sys.exit(1)

# ==========================================================
# ⚙️ 配置区域
# ==========================================================
# 生成时间戳文件夹名，格式：YYYYMMDD_HHMMSS
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# 根输出目录：../test/时间戳/
BASE_OUTPUT_DIR = os.path.join("..", "test", timestamp)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

OUTPUT_DIR = BASE_OUTPUT_DIR
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
IMG_DIR = os.path.join(OUTPUT_DIR, "images", "paths")
SUMMARY_IMG = os.path.join(OUTPUT_DIR, "images", "comparison_analysis.png")
CSV_FILE = os.path.join(OUTPUT_DIR, "experiment_results.csv")

MAP_WIDTH = 20
MAP_HEIGHT = 20
OBSTACLE_RATIO = 0.2

EXPERIMENT_SCENARIOS = [
    {
        "name": "纯小车 (1x1)",
        "configs": [{'w': 1, 'h': 1, 'ratio': 1.0}]
    },
    {
        "name": "混合方块 (1x1 + 2x2)",
        "configs": [
            {'w': 1, 'h': 1, 'ratio': 0.8},
            {'w': 2, 'h': 2, 'ratio': 0.2}
        ]
    },
    {
        "name": "混合长条 (1x1 + 1x2 + 2x1)",
        "configs": [
            {'w': 1, 'h': 1, 'ratio': 0.6},
            {'w': 1, 'h': 2, 'ratio': 0.2},
            {'w': 2, 'h': 1, 'ratio': 0.2}
        ]
    },
    {
        "name": "复杂异构 (1x1 + 2x2 + 3x1)",
        "configs": [
            {'w': 1, 'h': 1, 'ratio': 0.5},
            {'w': 2, 'h': 2, 'ratio': 0.25},
            {'w': 3, 'h': 1, 'ratio': 0.25}
        ]
    }
]

AGENT_COUNTS = list(range(4, 11)) 
REPEAT_TIMES = 50 
TIME_LIMIT_PER_RUN = 5.0
MAX_CBS_NODES = 3000
MAX_ATTEMPTS_PER_AGENT = 1500

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY_IMG), exist_ok=True)

log_filename = f"log_hetero_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_path = os.path.join(LOG_DIR, log_filename)

def log_message(msg, to_console=True):
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(full_msg + "\n")
    if to_console:
        print(full_msg)

# ==========================================================
# 可视化辅助
# ==========================================================
def plot_static_map(grid_map, agents, paths, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    draw_map(grid_map, ax=ax, show=False)
    draw_agents(agents, current_time=None, ax=ax, show=False)
    for agent in agents:
        path = paths.get(agent.global_id)
        if path:
            draw_path(agent, path=path, ax=ax, show=False)
    
    types = set([f"{a.shape[0]}x{a.shape[1]}" for a in agents])
    # 标题也使用中文字体
    ax.set_title(f"智能体数量={len(agents)} | 类型: {','.join(types)}", fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

# ==========================================================
# 核心逻辑
# ==========================================================

def run_single_trial(seed, total_agents, trial_id, scenario_name, agent_configs):
    random.seed(seed)
    np.random.seed(seed)
    start_time = time.time()
    
    try:
        g_map = GridMap(MAP_WIDTH, MAP_HEIGHT)
        g_map.create_random_obstacles(OBSTACLE_RATIO)
        if g_map.grid is None: raise Exception("Map Fail")

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
            
        passable_graphs_dict = {}
        for cls in agent_classes:
            pg = PassableGraph(category=cls.category)
            pg.build_from(grid_map=g_map, agent_class=cls)
            if not pg.V: 
                raise ValueError(f"Cat {cls.category} No Path")
            passable_graphs_dict[cls.category] = pg

        agents, _ = generate_tasks(
            agent_classes=agent_classes, counts=counts, grid_map=g_map,
            passable_graphs=passable_graphs_dict, existing_occupied=None,
            max_attempts_per_agent=MAX_ATTEMPTS_PER_AGENT
        )
        
        if len(agents) < int(total_agents * 0.8):
            raise Exception(f"Task Gen Fail ({len(agents)})")

        res_table = ReservationTable(bridge_cells=[])
        cr = [[0]*MAP_WIDTH for _ in range(MAP_HEIGHT)]
        cbs_solver = CBS(agents, passable_graphs_dict, res_table, cr)
        cbs_solver.time_limit = TIME_LIMIT_PER_RUN
        
        success, paths, stats = cbs_solver.search(interactive=False)
        elapsed = time.time() - start_time
        
        nodes = stats.get('nodes_expanded', 0) if stats else 0
        
        if elapsed > TIME_LIMIT_PER_RUN or nodes > MAX_CBS_NODES:
            success = False
            reason = "超时" if elapsed > TIME_LIMIT_PER_RUN else "节点过多"
            if stats: stats['reason'] = reason

        result = {
            'scenario': scenario_name,
            'seed': seed, 'total_agents': total_agents, 'trial_id': trial_id,
            'success': success, 'time': elapsed,
            'makespan': stats.get('makespan', -1) if stats and success else -1,
            'cost': stats.get('cost', -1) if stats and success else -1,
            'nodes': nodes,
            'generated_count': len(agents),
            'error': stats.get('reason', None) if not success else None,
            'image_path': None
        }
        
        if success and paths and trial_id == 1:
            safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "")
            img_name = f"{safe_name}_n{total_agents}_t1.png"
            img_path = os.path.join(IMG_DIR, img_name)
            try:
                plot_static_map(g_map, agents, paths, save_path=img_path)
                result['image_path'] = img_path
            except Exception: pass
            
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'scenario': scenario_name, 'seed': seed, 'total_agents': total_agents, 'trial_id': trial_id,
            'success': False, 'time': elapsed, 'error': str(e),
            'makespan': -1, 'cost': -1, 'nodes': 0, 'generated_count': 0, 'image_path': None
        }

# ==========================================================
# 主流程
# ==========================================================

def main():
    log_message("="*80)
    log_message("🚀 启动：异构性深度分析实验 (中文版)")
    log_message(f"场景数: {len(EXPERIMENT_SCENARIOS)} | 规模: {AGENT_COUNTS} | 重复: {REPEAT_TIMES}")
    log_message("="*80)

    all_results = []
    total_start = time.time()

    for scenario in EXPERIMENT_SCENARIOS:
        s_name = scenario['name']
        s_configs = scenario['configs']
        log_message(f"\n>>> 正在运行场景: [{s_name}]", to_console=True)
        
        for count in AGENT_COUNTS:
            succ_count = 0
            group_times = []
            
            for i in range(REPEAT_TIMES):
                seed = hash(f"{s_name}_{count}_{i}") % 100000 + i
                res = run_single_trial(seed, count, i+1, s_name, s_configs)
                
                if res:
                    all_results.append(res)
                    if res['success']:
                        succ_count += 1
                        group_times.append(res['time'])
                
                if (i+1) % 10 == 0:
                    print(f"  {s_name} | N={count}: {i+1}/{REPEAT_TIMES}...", end='\r')
            
            sr = succ_count / REPEAT_TIMES * 100
            avg_t = np.mean(group_times) if group_times else 0
            print(f"  {s_name} | N={count}: ✅ SR={sr:.1f}% | AvgT={avg_t:.3f}s")
            log_message(f"  [{s_name}] N={count} -> SR: {sr:.1f}%, Time: {avg_t:.3f}s")

    total_elapsed = time.time() - total_start
    log_message(f"\n🎉 所有场景测试完成！总耗时: {total_elapsed/60:.2f} 分钟")

    if all_results:
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['scenario', 'seed', 'total_agents', 'trial_id', 'success', 'time', 'makespan', 'cost', 'nodes', 'error', 'generated_count', 'image_path']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        plot_comparison_chart(all_results)
    else:
        log_message("⚠️ 无数据")

def plot_comparison_chart(results):
    """
    绘制多场景对比图 (已修复中文显示)
    """
    scenarios = list(set(r['scenario'] for r in results))
    counts = sorted(list(set(r['total_agents'] for r in results)))
    
    data = {s: {k: {'mean': [], 'std': []} for k in ['sr', 'time', 'makespan']} for s in scenarios}
    
    for s in scenarios:
        for c in counts:
            grp = [r for r in results if r['scenario'] == s and r['total_agents'] == c]
            succ = [r for r in grp if r['success']]
            
            sr_val = len(succ)/len(grp)*100 if grp else 0
            data[s]['sr']['mean'].append(sr_val)
            data[s]['sr']['std'].append(0)
            
            for k in ['time', 'makespan']:
                if succ:
                    vals = [r[k] for r in succ]
                    data[s][k]['mean'].append(np.mean(vals))
                    data[s][k]['std'].append(np.std(vals))
                else:
                    data[s][k]['mean'].append(0)
                    data[s][k]['std'].append(0)

    # 创建图表
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # 全局标题
    fig.suptitle('异构性对 CBS 性能的影响分析', fontsize=16, fontweight='bold')
    
    metrics = ['sr', 'time', 'makespan']
    # 【修复】这里全部使用中文标签
    titles = ['求解成功率 (%)', '平均求解时间 (秒)', '全局完成时间 (Makespan)']
    ylabels = ['成功率 (%)', '时间 (秒)', '时间步']
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))

    for i, m in enumerate(metrics):
        ax = axs[i]
        for idx, s in enumerate(scenarios):
            means = data[s][m]['mean']
            stds = data[s][m]['std']
            
            ax.errorbar(counts, means, yerr=stds, fmt='o-', label=s, color=colors[idx], 
                        capsize=4, linewidth=2, markersize=6, alpha=0.9)
        
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel('智能体数量', fontsize=11)
        ax.set_ylabel(ylabels[i], fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 图例也使用中文字体
        ax.legend(loc='best', fontsize=10, frameon=True)
        
        if m == 'sr': ax.set_ylim(0, 105)
        if m == 'time': ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(SUMMARY_IMG, dpi=300, bbox_inches='tight')
    log_message(f"📈 对比图表已保存: {SUMMARY_IMG}")
    print(f"\n[完成] 请查看 {SUMMARY_IMG} (应包含清晰的中文标签)")
    plt.show()

if __name__ == "__main__":
    main()