"""
主控脚本：交互式输入参数，测试无冲突场景下的多智能体路径规划
"""
import sys
print("Script started", file=sys.stderr)

import random
import matplotlib.pyplot as plt
import sys

from gridmap import GridMap
from sh_agent import AgentClass
from passable_graph import PassableGraph
from task_generator import generate_tasks, filter_feasible_tasks
from priority import sort_tasks_by_priority
from congestion_coefficient import compute_cr
from reservation_table import ReservationTable
from astar import astar
from visualization import draw_map, draw_agents, draw_path, animate_solution


def get_positive_int(prompt, default=None):
    """获取正整数输入"""
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

def get_float_in_range(prompt, low, high, default=None):
    """获取指定范围内的浮点数"""
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

def get_non_negative_int(prompt, default=None):
    """获取非负整数"""
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

def main():
    try:
        print("=" * 50)
        print("多智能体路径规划测试脚本（无冲突场景）")
        print("=" * 50)

        # ==================== 交互式输入参数 ====================
        print("\n请配置地图参数：")
        MAP_WIDTH = get_positive_int("地图宽度 (列数) [默认10]: ", default=10)
        MAP_HEIGHT = get_positive_int("地图高度 (行数) [默认10]: ", default=10)
        OBSTACLE_RATIO = get_float_in_range("障碍物占比 [0,1) [默认0.2]: ", 0.0, 1.0, default=0.2)

        # 智能体类别配置
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

        # 汇总显示
        print("\n" + "=" * 50)
        print("输入参数汇总：")
        print(f"地图尺寸: {MAP_WIDTH} x {MAP_HEIGHT}")
        print(f"障碍物占比: {OBSTACLE_RATIO}")
        for cls, cnt in zip(agent_classes, counts):
            print(f"类别 {cls.category}: 尺寸 {cls.width}x{cls.height}, 数量 {cnt}")
        print("=" * 50)

        # 设置随机种子（可选）
        seed = input("\n输入随机种子 (直接回车使用默认42): ")
        if seed:
            random.seed(int(seed))
        else:
            random.seed(42)

        # ==================== 1. 创建地图 ====================
        print("\n[1] 创建地图...")
        grid = GridMap(MAP_WIDTH, MAP_HEIGHT)
        obstacles = grid.create_random_obstacles(OBSTACLE_RATIO)
        print(f"    地图尺寸: {grid.rows}行 x {grid.cols}列")
        print(f"    障碍物数量: {len(obstacles)}")

        # ==================== 2. 构建可通行子图 ====================
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

        # ==================== 3. 生成任务 ====================
        print("\n[3] 生成任务...")
        tasks = generate_tasks(agent_classes, counts, grid, passable_graphs)
        print(f"    初始生成任务数: {len(tasks)}")

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
            print(f"    智能体 {t.id} (类别 {t.agent_class.category}, 尺寸 {t.agent_class.width}x{t.agent_class.height}, "
                  f"优先级 {prio}, 距离 {dist}) 起点 {t.start} -> 终点 {t.goal}")

        # ==================== 5. 计算拥堵系数 ====================
        print("\n[5] 计算拥堵系数...")
        cr = compute_cr(grid, passable_graphs)
        print("    拥堵系数矩阵 (前5行):")
        for y in range(min(5, grid.rows)):
            row = [cr[x][y] for x in range(grid.cols)]
            print(f"      y={y}: {row}")

        # ==================== 6. 初始化预约表 ====================
        print("\n[6] 初始化预约表...")
        all_bridges = set()
        for pg in passable_graphs.values():
            all_bridges.update(pg.bridges)
        reservation = ReservationTable(bridge_cells=all_bridges)

        # ==================== 7. 按顺序规划路径 ====================
        print("\n[7] 开始规划路径...")
        for task in tasks:
            print(f"\n    规划智能体 {task.id} (类别 {task.agent_class.category})...")
            path = astar(
                agent_instance=task,
                passable_graph=passable_graphs[task.agent_class.category],
                reservation_table=reservation,
                cr=cr,
                weight_cr=0.1,
                weight_bridge=10.0,
                weight_res=1.0
            )
            if path is None:
                print(f"    警告：智能体 {task.id} 无法找到可行路径！")
                task.set_path([])
            else:
                task.set_path(path)
                print(f"    路径长度: {len(path)} (包含起点)")
                reservation.add(task.id, path, task.agent_class)

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
                print(f"智能体 {task.id}: 路径 {task.path}, 到达时间 {task.arrival_time}")
            else:
                print(f"智能体 {task.id}: 无路径")
        print(f"总路径代价 (时间步之和): {total_cost}")
        print(f"全局完成时间: {max_time}")

        # ==================== 9. 可视化 ====================
        print("\n[9] 显示静态地图和路径 (请关闭图形窗口以继续)...")
        # 静态图
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        draw_map(grid, ax, show=False)
        draw_agents(tasks, current_time=None, ax=ax, show=False)
        for task in tasks:
            if task.path:
                draw_path(task, ax=ax, show=False)
        plt.title("Static Map with Start, Goal, and Paths")
        plt.show()

        # 动画
        if any(task.path for task in tasks):
            print("\n生成动画 (请关闭动画窗口以结束程序)...")
            animate_solution(tasks, grid, interval=500, save_path=None)
        else:
            print("\n没有有效路径，无法生成动画。")

        print("\n程序正常结束。")

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()