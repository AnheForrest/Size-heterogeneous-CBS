"""
主控脚本：交互式输入参数，测试多智能体路径规划（支持冲突解决）
"""

import sys
print("Script started", file=sys.stderr)

import random
import matplotlib.pyplot as plt

from gridmap import GridMap
from sh_agent import AgentClass
from passable_graph import PassableGraph
from task_generator import generate_tasks, filter_feasible_tasks, retry_failed_tasks
from priority import sort_tasks_by_priority
from congestion_coefficient import compute_cr
from reservation_table import ReservationTable
from astar import astar
from cbs import CBS                     # 新增导入
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
        print("多智能体路径规划测试脚本（支持冲突解决）")
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

        # ==================== 3. 生成任务（考虑起点/终点占用冲突） ====================
        print("\n[3] 生成任务...")
        tasks, failed_requests = generate_tasks(
            agent_classes,
            counts,
            grid,                       # 第四个参数：grid_map
            passable_graphs,
            existing_occupied=None,
            max_attempts_per_agent=1000
        )
        print(f"    成功生成任务数: {len(tasks)}")
        if failed_requests:
            print(f"    有 {len(failed_requests)} 个智能体因起点/终点冲突或不可达而生成失败。")
            # 询问是否重试
            resp = input("是否重新生成这些失败的任务？(y/n): ").strip().lower()
            if resp == 'y':
                # 收集当前已成功任务的占用栅格
                occupied = set()
                for t in tasks:
                    occupied.update(t.agent_class.get_occupied_cells(t.start))
                    occupied.update(t.agent_class.get_occupied_cells(t.goal))
                # 尝试重新生成失败的任务
                new_tasks = retry_failed_tasks(
                    failed_requests,
                    agent_classes,
                    passable_graphs,
                    existing_occupied=occupied,   # occupied 会被更新
                    max_attempts=10
                )
                tasks.extend(new_tasks)
                print(f"    重新生成成功，新增 {len(new_tasks)} 个任务。")
            else:
                print("    已忽略失败的任务，继续使用已生成的任务。")
        else:
            print("    所有任务生成成功，无失败请求。")

        # 可选：再次过滤（确保连通性，但生成时已保证）
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
        # 创建一个仅包含桥信息的空预约表（供 CBS 内部使用）
        empty_reservation = ReservationTable(bridge_cells=all_bridges)

        # ==================== 7. 调用 CBS 求解 ====================
        print("\n[7] 启动 CBS 搜索...")
        cbs = CBS(
            agents=tasks,
            passable_graphs=passable_graphs,
            reservation_table=empty_reservation,   # 空预约表（仅含桥信息）
            cr=cr
        )
        solution = cbs.search()

        if solution is None:
            print("CBS 未找到可行解，程序退出。")
            return
        else:
            print("CBS 找到解，正在处理结果...")
            # 将解路径赋给对应的智能体
            for agent in tasks:
                if agent.global_id in solution:
                    agent.set_path(solution[agent.global_id])
                else:
                    agent.set_path([])   # 理论上不会发生

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