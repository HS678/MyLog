import random


def reservoir_sampling_correct(stream, k):
    """正确的蓄水池抽样实现"""
    reservoir = stream[:k]  # 前k个元素直接进入

    for i in range(k, len(stream)):
        j = random.randint(0, i)
        if j < k:
            reservoir[j] = stream[i]

    return reservoir


def calculate_probability_correctly(stream, k, trials=10000):
    """正确计算概率的测试函数"""
    # 记录每个元素被选中的实验次数
    element_selected_count = {element: 0 for element in stream}

    for _ in range(trials):
        sample = reservoir_sampling_correct(stream, k)
        # 检查每个元素是否在这次实验中被选中
        for element in stream:
            if element in sample:
                element_selected_count[element] += 1

    n = len(stream)
    theoretical_prob = k / n

    print(f"=== 正确概率计算 ===")
    print(f"数据流: {stream}")
    print(f"k={k}, n={n}, 实验次数: {trials}")
    print(f"理论概率 (单次实验): {theoretical_prob:.3f}")
    print("实际频率 (单次实验):")

    for element in stream:
        count = element_selected_count[element]
        experimental_freq = count / trials
        deviation = abs(experimental_freq - theoretical_prob)
        print(f"  {element}: {experimental_freq:.3f} (偏差: {deviation:.3f}, 在{count}次实验中被选中)")

    # 验证：所有元素被选中的总次数应该接近 trials * k
    total_selections = sum(element_selected_count.values())
    expected_total = trials * k
    print(f"\n验证:")
    print(f"  实际总选中次数: {total_selections}")
    print(f"  期望总选中次数: {expected_total}")
    print(f"  差异: {abs(total_selections - expected_total)}")


# 运行测试
calculate_probability_correctly(['A', 'B', 'C', 'D'], k=2, trials=10000)