"""
主控脚本
"""

import sys
print("Script started", file=sys.stderr)

import random
import matplotlib.pyplot as plt
import os
from datetime import datetime

from gridmap import GridMap
from sh_agent import AgentClass
from passable_graph import PassableGraph
from task_generator import generate_tasks, filter_feasible_tasks, retry_failed_tasks
from priority import sort_tasks_by_priority
from congestion_coefficient import compute_cr
from reservation_table import ReservationTable
from astar import astar
from cbs import CBS
from visualization import draw_map, draw_agents, draw_path, animate_solution

"""------------------------0.交互输入参数------------------------"""
#0.1.获取正整数输入
def get_positive_int(prompt, default=None):
    while True:
        val = input(prompt)
        if not val and default is not None:
            return default
        try:
            ival = int(val)
            if ival > 0:
                return ival
            else:
                print("请输入正整数。")
        except ValueError:
            print("请输入整数。")

#0.2.获取指定范围内的浮点数
def get_float_in_range(prompt, low, high, default=None):
    while True:
        val = input(prompt)
        if not val and default is not None:
            return default
        try:
            fval = float(val)
            if low <= fval < high:
                return fval
            else:
                print(f"请输入 [{low}, {high}) 之间的数。")
        except ValueError:
            print("请输入浮点数。")

#0.3.获取非负整数
def get_non_negative_int(prompt, default=None):
    while True:
        val = input(prompt)
        if not val and default is not None:
            return default
        try:
            ival = int(val)
            if ival >= 0:
                return ival
            else:
                print("请输入非负整数。")
        except ValueError:
            print("请输入整数。")

#0.4.获取是/否输入
def get_yes_no(prompt, default='n'):
    while True:
        val = input(prompt).strip().lower()
        if not val:
            return default == 'y'
        if val in ['y', 'yes', '是']:
            return True
        elif val in ['n', 'no', '否']:
            return False
        else:
            print("请输入 y 或 n。")

#0.5.生成带时间戳的输出文件名
def generate_output_filename(prefix, extension):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def main():
    try:
        print("=" * 50)
        print("尺寸异构多智能体路径规划主控脚本")
        print("=" * 50)

        #==================== 交互式输入参数 ====================
        print("\n请配置地图参数：")
        MAP_WIDTH = get_positive_int("地图宽度 (列数) [默认10]: ", default=10)
        MAP_HEIGHT = get_positive_int("地图高度 (行数) [默认10]: ", default=10)
        OBSTACLE_RATIO = get_float_in_range("障碍物占比 [0,1) [默认0.2]: ", 0.0, 1.0, default=0.2)

        #智能体类别配置
        num_categories = get_positive_int("\n智能体类别数 [默认2]: ", default=2)
        agent_classes = []
        counts = []
        print("\n请配置每类智能体的尺寸和数量：")
        for cat in range(1, num_categories + 1):
            print(f"\n--- 类别 {cat} ---")
            width = get_positive_int(f"  宽度 (格子数) [默认2]: ", default=2)
            height = get_positive_int(f"  高度 (格子数) [默认2]: ", default=2)
            count = get_non_negative_int(f"  该类智能体数量 [默认1]: ", default=1)
            agent_classes.append(AgentClass(category=cat, width=width, height=height))
            counts.append(count)

        #汇总显示
        print("\n" + "=" * 50)
        print("输入参数汇总：")
        print(f"地图尺寸: {MAP_WIDTH} x {MAP_HEIGHT}")
        print(f"障碍物占比: {OBSTACLE_RATIO}")
        for cls, cnt in zip(agent_classes, counts):
            print(f"类别 {cls.category}: 尺寸 {cls.width}x{cls.height}, 数量 {cnt}")
        print("=" * 50)

        #设置随机种子
        seed = input("\n输入随机种子 (直接回车使用默认10): ")
        if seed:
            random.seed(int(seed))
        else:
            random.seed(10)

        #==================== 1. 创建地图 ====================
        print("\n[1] 创建地图...")
        grid = GridMap(MAP_WIDTH, MAP_HEIGHT)
        obstacles = grid.create_random_obstacles(OBSTACLE_RATIO)
        print(f"    地图尺寸: {grid.rows}行 x {grid.cols}列")
        print(f"    障碍物数量: {len(obstacles)}")

        #==================== 2. 构建可通行子图 ====================
        print("\n[2] 构建可通行子图...")
        passable_graphs = {}
        for cls in agent_classes:
            pg = PassableGraph(cls.category)
            pg.build_from(grid, cls)
            pg.width = cls.width
            pg.height = cls.height
            pg.find_bridges(cls)
            passable_graphs[cls.category] = pg
            print(f"    类别 {cls.category}: |V|={len(pg.V)}, |E|={sum(len(v) for v in pg.E.values())//2}, 桥栅格数={len(pg.bridges)}")

        # ==================== 3. 生成任务（考虑起点/终点占用冲突） ====================
        print("\n[3] 生成任务...")
        tasks, failed_requests = generate_tasks(
            agent_classes,
            counts,
            grid,
            passable_graphs,
            existing_occupied=None,
            max_attempts_per_agent=1000
        )
        print(f"    成功生成任务数: {len(tasks)}")
        if failed_requests:
            print(f"    有 {len(failed_requests)} 个智能体因起点/终点冲突或不可达而生成失败。")
            resp = input("是否重新生成这些失败的任务？(y/n): ").strip().lower()
            if resp == 'y':
                occupied = set()
                for t in tasks:
                    occupied.update(t.agent_class.get_occupied_cells(t.start))
                    occupied.update(t.agent_class.get_occupied_cells(t.goal))
                new_tasks = retry_failed_tasks(
                    failed_requests,
                    agent_classes,
                    passable_graphs,
                    existing_occupied=occupied,
                    max_attempts=10
                )
                tasks.extend(new_tasks)
                print(f"    重新生成成功，新增 {len(new_tasks)} 个任务。")
            else:
                print("    已忽略失败的任务，继续使用已生成的任务。")
        else:
            print("    所有任务生成成功，无失败请求。")

        tasks = filter_feasible_tasks(tasks, passable_graphs)
        print(f"    过滤后剩余任务数: {len(tasks)}")

        if not tasks:
            print("没有可行任务，程序退出。")
            return

        # ==================== 4. 优先级排序 ====================
        print("\n[4] 优先级排序...")
        tasks = sort_tasks_by_priority(tasks)
        for t in tasks:
            prio = t.agent_class.width + t.agent_class.height
            dist = abs(t.start[0]-t.goal[0]) + abs(t.start[1]-t.goal[1])
            print(f"    智能体 {t.id_str} (类别 {t.agent_class.category}, 尺寸 {t.agent_class.width}x{t.agent_class.height}, "
                  f"优先级 {prio}, 距离 {dist}) 起点 {t.start} -> 终点 {t.goal}")

        # ==================== 5. 计算拥堵系数 ====================
        print("\n[5] 计算拥堵系数...")
        cr = compute_cr(grid, passable_graphs)
        print("    拥堵系数矩阵 (前5行):")
        for y in range(min(5, grid.rows)):
            row = [cr[x][y] for x in range(grid.cols)]
            print(f"      y={y}: {row}")

        # ==================== 6. 初始化桥栅格集合和空预约表 ====================
        print("\n[6] 准备 CBS 所需数据...")
        all_bridges = set()
        for pg in passable_graphs.values():
            all_bridges.update(pg.bridges)
        empty_reservation = ReservationTable(bridge_cells=all_bridges)

        cbs = CBS(
            agents=tasks,
            passable_graphs=passable_graphs,
            reservation_table=empty_reservation,
            cr=cr
        )

        interactive_mode = input("\n是否进入交互调试模式？(y/n, 默认 n): ").strip().lower() == 'y'

        # ==================== 7. 调用 CBS 求解 ====================
        print("\n[7] 启动 CBS 搜索...")
        result = cbs.search(interactive=interactive_mode)

        if isinstance(result, tuple) and len(result) == 3:
            success, solution, stats = result
        else:
            success = result is not None
            solution = result
            stats = None

        if not success or solution is None:
            print("CBS 未找到可行解，程序退出。")
            return
        else:
            print("CBS 找到解，正在处理结果...")
            for agent in tasks:
                if agent.global_id in solution:
                    agent.set_path(solution[agent.global_id])
                else:
                    agent.set_path([])

        # ==================== 8. 结果输出 ====================
        print("\n" + "=" * 50)
        print("最终结果")
        print("=" * 50)
        total_cost = 0
        max_time = 0
        for task in tasks:
            if task.path:
                cost = len(task.path) - 1
                total_cost += cost
                max_time = max(max_time, len(task.path)-1)
                print(f"智能体 {task.id_str}: 路径 {task.path}, 到达时间 {task.arrival_time}")
            else:
                print(f"智能体 {task.id_str}: 无路径")
        print(f"总路径代价 (时间步之和): {total_cost}")
        print(f"全局完成时间: {max_time}")

        # ==================== 9. 可视化 ====================
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\n    已创建输出目录: {output_dir}/")

        # --- 9.1 静态图 ---
        print("\n[9.1] 生成静态路径图...")
        save_static = get_yes_no("    是否保存静态图到本地？(y/n, 默认 n): ", default='n')
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        draw_map(grid, ax, show=False)
        draw_agents(tasks, current_time=None, ax=ax, show=False)
        for task in tasks:
            if task.path:
                draw_path(task, ax=ax, show=False)
        plt.title(f"Static Map with Start, Goal, and Paths\nAgents: {len(tasks)}, Makespan: {max_time}")
        plt.tight_layout()
        
        if save_static:
            static_filename = os.path.join(output_dir, generate_output_filename("static_map", "png"))
            plt.savefig(static_filename, dpi=300, bbox_inches='tight')
            print(f"    静态图已保存：{static_filename}")
        
        show_static = get_yes_no("    是否在屏幕上显示静态图？(y/n, 默认 y): ", default='y')
        if show_static:
            plt.show()
        else:
            plt.close(fig)

        # --- 9.2 动画 ---
        if any(task.path for task in tasks):
            print("\n[9.2] 生成动画...")
            save_animation = get_yes_no("    是否保存动画到本地？(y/n, 默认 n): ", default='n')
            
            animation_file = None
            if save_animation:
                print("    可选格式：mp4 (需要ffmpeg), gif, webm")
                anim_format = input("    输入动画格式 (默认gif): ").strip().lower()
                if not anim_format:
                    anim_format = "gif"
                animation_file = os.path.join(output_dir, generate_output_filename("animation", anim_format))
                print(f"    动画将保存为：{animation_file}")
            
            interval = get_positive_int("    动画帧间隔 (毫秒) [默认500]: ", default=500)
            
            # 修复：移除 show 参数，save_path=None 时自动显示
            animate_solution(
                tasks, 
                grid, 
                interval=interval, 
                save_path=animation_file
            )
            
            if animation_file and os.path.exists(animation_file):
                print(f"    动画已保存：{animation_file}")
        else:
            print("\n没有有效路径，无法生成动画。")

        # ==================== 10. 完成 ====================
        print("\n" + "=" * 50)
        print("程序正常结束。")
        print(f"输出目录：{os.path.abspath(output_dir)}/")
        print("=" * 50)

    except Exception as e:
        print(f"\n发生错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()