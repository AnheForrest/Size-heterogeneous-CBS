"""
可视化模块
功能：
    1.draw_map: 绘制基底地图
    2.draw_agents: 绘制智能体的起点、终点、当前位置
    3.draw_path: 绘制单个智能体的路径
    4.animate_solution: 动画展示整个调度过程
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple

#颜色定义
COLOR_FREE = 'white'           #可通行栅格背景
COLOR_OBSTACLE = 'black'        #障碍物
COLOR_AGENTS = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
                'pink', 'gray', 'olive', 'cyan']  #多智能体颜色循环

def draw_map(grid_map, ax=None, show=True):
    """
    绘制基底地图。

    :param grid_map: GridMap 对象，包含 cols, rows, grid 属性（grid[y][x] 为 0/1）
    :param ax: matplotlib 坐标轴，若为 None 则新建
    :param show: 是否立即显示图像
    :return: 坐标轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cols, rows = grid_map.cols, grid_map.rows

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(cols+1))
    ax.set_yticks(range(rows+1))
    ax.grid(True, linestyle='-', color='gray', linewidth=0.5)

    for y in range(rows):
        for x in range(cols):
            if grid_map.grid[y][x] == 1:
                rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=COLOR_OBSTACLE)
                ax.add_patch(rect)

    if show:
        plt.show()
    return ax


def draw_agents(agents, current_time: Optional[int] = None, ax=None, show=True):
    """
    绘制智能体的起点、终点、当前位置（如果指定了 current_time）。

    :param agents: AgentInstance 对象列表
    :param current_time: 当前时间步，若提供则绘制该时刻每个智能体的位置；否则只绘制起点和终点
    :param ax: matplotlib 坐标轴
    :param show: 是否立即显示图像
    :return: 坐标轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for i, agent in enumerate(agents):
        color = COLOR_AGENTS[i % len(COLOR_AGENTS)]
        w, h = agent.agent_class.width, agent.agent_class.height

        #绘制起点（半透明填充 + "S"）
        start_rect = patches.Rectangle(agent.start, w, h, linewidth=1,
                                        edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(start_rect)
        ax.text(agent.start[0] + w/2, agent.start[1] + h/2, 'S',
                color='white', ha='center', va='center', fontsize=8, weight='bold')

        #绘制终点（半透明填充 + "G"）
        goal_rect = patches.Rectangle(agent.goal, w, h, linewidth=1,
                                       edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(goal_rect)
        ax.text(agent.goal[0] + w/2, agent.goal[1] + h/2, 'G',
                color='white', ha='center', va='center', fontsize=8, weight='bold')

        #如果指定了当前时间，绘制智能体当前位置及其编号
        if current_time is not None and agent.path is not None:
            if current_time < len(agent.path):
                pos = agent.path[current_time]
                agent_rect = patches.Rectangle(pos, w, h, linewidth=1,
                                                edgecolor=color, facecolor=color, alpha=0.7)
                ax.add_patch(agent_rect)
                ax.text(pos[0] + w/2, pos[1] + h/2, agent.id_str,
                        color='white', ha='center', va='center', fontsize=8, weight='bold')

    if show:
        plt.show()
    return ax


def draw_path(agent, path=None, ax=None, show=True):
    """
    绘制单个智能体的路径（用线连接）。

    :param agent: AgentInstance 对象
    :param path: 路径坐标列表，若为 None 则使用 agent.path
    :param ax: matplotlib 坐标轴
    :param show: 是否立即显示图像
    :return: 坐标轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    if path is None:
        path = agent.path
    if path is None or len(path) == 0:
        return ax

    #提取路径的中心点（矩形中心），用于连线
    w2 = agent.agent_class.width / 2
    h2 = agent.agent_class.height / 2
    centers = [(x + w2, y + h2) for (x, y) in path]

    xs, ys = zip(*centers)
    #颜色基于agent在全局列表中的索引？但此处无法获取列表，故使用agent.global_id的哈希值或默认索引。
    #为简单起见，使用agent.id_str的哈希值映射到颜色列表，保证同一智能体颜色一致。
    color_idx = hash(agent.id_str) % len(COLOR_AGENTS)
    color = COLOR_AGENTS[color_idx]
    ax.plot(xs, ys, color=color, linestyle='-', linewidth=2, alpha=0.7, marker='.', markersize=4)

    if show:
        plt.show()
    return ax


def animate_solution(agents, grid_map, interval=200, save_path=None):
    """
    动画展示整个调度过程。

    :param agents: AgentInstance 对象列表，每个对象需有 path 属性
    :param grid_map: GridMap 对象
    :param interval: 每帧间隔毫秒
    :param save_path: 如果提供，保存动画为文件（如 .gif 或 .mp4）
    :return: 动画对象
    """
    #确定最大时间步
    max_time = 0
    for agent in agents:
        if agent.path is not None:
            max_time = max(max_time, len(agent.path))
    if max_time == 0:
        print("没有路径数据，无法生成动画。")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    draw_map(grid_map, ax, show=False)

    #绘制所有智能体的起点和终点
    for i, agent in enumerate(agents):
        color = COLOR_AGENTS[i % len(COLOR_AGENTS)]
        w, h = agent.agent_class.width, agent.agent_class.height

        #起点
        start_rect = patches.Rectangle(agent.start, w, h, linewidth=1,
                                        edgecolor=color, facecolor=color, alpha=0.1)
        ax.add_patch(start_rect)
        ax.text(agent.start[0] + w/2, agent.start[1] + h/2, 'S',
                color='white', ha='center', va='center', fontsize=8, weight='bold')

        # 终点
        goal_rect = patches.Rectangle(agent.goal, w, h, linewidth=1,
                                       edgecolor=color, facecolor=color, alpha=0.1)
        ax.add_patch(goal_rect)
        ax.text(agent.goal[0] + w/2, agent.goal[1] + h/2, 'G',
                color='white', ha='center', va='center', fontsize=8, weight='bold')

    #用于存储智能体当前位置矩形的引用
    agent_rects = []
    #用于存储智能体编号文本的引用
    agent_texts = []
    for i, agent in enumerate(agents):
        color = COLOR_AGENTS[i % len(COLOR_AGENTS)]
        w, h = agent.agent_class.width, agent.agent_class.height
        #初始位置设为起点
        rect = patches.Rectangle(agent.start, w, h, linewidth=1,
                                  edgecolor=color, facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        agent_rects.append(rect)

        #编号文本
        txt = ax.text(agent.start[0] + w/2, agent.start[1] + h/2, agent.id_str,
                      color='white', ha='center', va='center', fontsize=8, weight='bold')
        agent_texts.append(txt)

    #时间文本
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def init():
        return agent_rects + agent_texts + [time_text]

    def update(frame):
        for i, agent in enumerate(agents):
            if agent.path is not None and frame < len(agent.path):
                pos = agent.path[frame]
                agent_rects[i].set_xy(pos)
                # 更新文本位置
                w = agent.agent_class.width
                h = agent.agent_class.height
                agent_texts[i].set_position((pos[0] + w/2, pos[1] + h/2))
        time_text.set_text(f'Time: {frame}')
        return agent_rects + agent_texts + [time_text]

    ani = FuncAnimation(fig, update, frames=range(max_time), init_func=init,
                        interval=interval, blit=True, repeat=True)

    if save_path:
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=1000//interval)
        else:
            ani.save(save_path, writer='ffmpeg', fps=1000//interval)
    else:
        plt.show()
    return ani