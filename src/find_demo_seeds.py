import subprocess
import sys

# 【固定：你论文里唯一稳定的演示参数】
# 地图 20×20，障碍0.2，3类智能体 1×1 / 1×2 / 2×1，每类2个，共6个

seed = 1


print("===== 手动测试：直接跑正确输入，看是否正常求解 =====")
print("输入顺序：宽度 高度 障碍 类别数 w h 数量 w h 数量 种子 n n\n")

while True:
    print(f"\n===== 测试种子：{seed} =====")
    p = subprocess.Popen(
        [sys.executable, "main_cbs.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="gbk"
    )

    # 这是 100% 正确、和你手工输入完全一样的顺序
    commands = [
        "20", "20", "0.1",
        "3",
        "1","1","3",
        "1","2","2",
        "2","2","2",
        str(seed),
        "n","n"
    ]

    try:
        out, _ = p.communicate("\n".join(commands)+"\n", timeout=10)
        print(out)
    except:
        print("超时/错误")

    seed += 1
    input("按回车继续下一个种子...")
