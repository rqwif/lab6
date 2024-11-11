import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Матриця виграшів
F = np.array([
    [3, 8, 7, 9],
    [5, 6, 8, 3],
    [4, 9, 4, 9],
    [6, 4, 5, 4]
])

# Ймовірності для кожного стану
p = np.array([0.25, 0.15, 0.4, 0.2])

# Критерій Вальда (максимінний критерій)
def wald_criterion(F):
    min_values = np.min(F, axis=1)  # Мінімальні значення для кожної стратегії
    wald_value = np.max(min_values)  # Максимум серед мінімумів
    optimal_strategy = np.argmax(min_values)  # Стратегія з максимальним мінімальним значенням
    return optimal_strategy, wald_value

# Критерій мінімізації середньоквадратичного відхилення
def min_variance_criterion(F, p):
    expected_values = np.dot(F, p)  # Очікувані значення для кожної стратегії
    variances = np.sum(((F.T - expected_values) ** 2) * p, axis=0)  # Дисперсії для кожної стратегії
    min_variance_strategy = np.argmin(variances)  # Стратегія з мінімальною дисперсією
    return min_variance_strategy, variances

# Знаходження оптимального рішення за критерієм Вальда
optimal_wald_strategy, wald_value = wald_criterion(F)
print(f"Оптимальна стратегія за критерієм Вальда: {optimal_wald_strategy + 1} (значення: {wald_value})")

# Знаходження оптимального рішення за критерієм мінімізації середньоквадратичного відхилення
optimal_variance_strategy, variances = min_variance_criterion(F, p)
print(f"Оптимальна стратегія за мінімальним середньоквадратичним відхиленням: {optimal_variance_strategy + 1}")
print("Дисперсії для кожної стратегії:", variances)

# Побудова ієрархічної схеми
def draw_box(ax, text, x, y, width=0.25, height=0.1, fontsize=10, with_text=True):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", edgecolor="black", facecolor="lightgray")
    ax.add_patch(box)
    if with_text:
        ax.text(x + width/2, y + height/2, text, ha="center", va="center", fontsize=fontsize)

fig, ax = plt.subplots(figsize=(12, 10))
ax.axis("off")

# Верхній рівень (Ціль)
draw_box(ax, "", 0.4, 0.8, width=0.3, height=0.15, fontsize=12, with_text=False)
ax.text(0.55, 0.88, "Оптимальне рішення\n(Ціль)", ha="center", va="center", fontsize=12)

# Другий рівень (Критерії)
draw_box(ax, "", 0.2, 0.6, width=0.3, height=0.12, with_text=False)
draw_box(ax, "", 0.6, 0.6, width=0.3, height=0.12, with_text=False)
ax.text(0.35, 0.66, "Мінімальне середньоквадратичне\nвідхилення", ha="center", va="center", fontsize=10)
ax.text(0.75, 0.66, "Критерій Вальда", ha="center", va="center", fontsize=10)

# Третій рівень (Стратегії)
strategies = ["Стратегія 1", "Стратегія 2", "Стратегія 3", "Стратегія 4"]
for i, strategy in enumerate(strategies):
    draw_box(ax, "", 0.1 + i*0.2, 0.4, width=0.2, height=0.1, with_text=False)
    ax.text(0.2 + i*0.2, 0.42, strategy, ha="center", va="center", fontsize=10)

# З'єднання стрілками між рівнями
for i in range(4):
    ax.annotate("", xy=(0.35, 0.62), xytext=(0.2 + i*0.2, 0.45), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.75, 0.62), xytext=(0.2 + i*0.2, 0.45), arrowprops=dict(arrowstyle="->"))

# З'єднання верхнього рівня з критеріями
ax.annotate("", xy=(0.55, 0.75), xytext=(0.35, 0.63), arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(0.65, 0.75), xytext=(0.75, 0.63), arrowprops=dict(arrowstyle="->"))

plt.show()
