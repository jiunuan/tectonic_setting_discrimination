import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# 中文注释：现代玄武岩全球分布图读取当前项目合并总表和外部世界底图资产。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import COMBINED_CSV, ZENODO_MODERN_CSV, WORLD_BASEMAP_PNG, MODERN_DISTRIBUTION_MAP_PNG
from matplotlib.image import imread

# 读取CSV文件
# 中文注释：现代样品点位读取当前项目合并总表。
file_path = str(ZENODO_MODERN_CSV if Path(ZENODO_MODERN_CSV).exists() else COMBINED_CSV)
df = pd.read_csv(file_path, encoding='utf-8')

# 提取经纬度数据和构造环境数据
latitudes = df['latitude'].tolist()
longitudes = df['longitude'].tolist()
tectonic_settings = df['TECTONIC SETTING'].tolist()

# 定义不同构造环境的颜色
# tectonic_colors = {
#     'BACK-ARC_BASIN': '#32CD32',  # 鲜绿色
#     'CONTINENTAL_RIFT': '#4682B4',  # 钢蓝色
#     'Continental arc': '#FF4500',  # 橙红色
#     'Island arc': '#FFD700',  # 金色
#     'Intra-oceanic arc': '#6A5ACD',  # 板岩蓝
#     'CONTINENTAL FLOOD BASALT': '#8B4513',  # 馅饼褐色
#     'OCEAN ISLAND': '#000000',  # 黑色
#     'OCEANIC PLATEAU': '#800080',  # 紫色
#     'SPREADING_CENTER': '#FFA500'  # 橙色
# }
# tectonic_colors = {
#     'BACK-ARC_BASIN': '#00FA9A',  # 中亮的春绿色
#     'CONTINENTAL_RIFT': '#1E90FF',  # 道奇蓝
#     'Continental arc': '#FF4500',  # 橙红色
#     'Island arc': '#FFD700',  # 金色
#     'Intra-oceanic arc': '#483D8B',  # 暗蓝紫色
#     'CONTINENTAL FLOOD BASALT': '#CD853F',  # 沙褐色
#     'OCEAN ISLAND': '#2F4F4F',  # 深石板灰
#     'OCEANIC PLATEAU': '#FF00FF',  # 亮紫色
#     'SPREADING_CENTER': '#FF8C00'  # 深橙色
# }
tectonic_colors = {
    # 海洋构造（偏暖色调，与蓝色海洋形成对比）
    'SPREADING_CENTER': '#FFCC80',  # 浅橙色 - 降低饱和度减少视觉压迫
    'OCEAN ISLAND': '#00CED1',  # 深青色 - 在海洋和图例白色背景上均可见
    'OCEANIC PLATEAU': '#FF69B4',  # 粉红色 - 主要分布在海洋区域
    'BACK-ARC_BASIN': '#32CD32',  # 柠檬绿 - 与周围海洋和弧形成对比

    # 弧构造（鲜艳色系，需要突出显示）
    'Continental arc': '#FF0000',  # 鲜红色 - 在褐色山地地形上醒目
    'Island arc': '#FFD700',  # 金色 - 在蓝色海洋和绿色陆地上都清晰
    'Intra-oceanic arc': '#8B008B',  # 深紫色 - 主要在深色海洋区域

    # 陆地构造（考虑陆地地形色系）
    'CONTINENTAL FLOOD BASALT': '#8C510A',  # 砖红色 - 与陆地棕褐色地形形成明显对比
    'CONTINENTAL_RIFT': '#1E90FF'  # 道奇蓝 - 在陆地色系上清晰可见
}
# tectonic_colors = {
#     'BACK-ARC_BASIN': '#FF1493',        # 深粉红
#     'CONTINENTAL_RIFT': '#00FFFF',      # 青色
#     'Continental arc': '#FF4500',       # 橙红色
#     'Island arc': '#FFD700',            # 金色
#     'Intra-oceanic arc': '#FF00FF',     # 洋红
#     'CONTINENTAL FLOOD BASALT': '#8B4513',  # 马鞍棕色
#     'OCEAN ISLAND': '#FFFFFF',          # 白色
#     'OCEANIC PLATEAU': '#7FFF00',       # 查特鲁斯绿
#     'SPREADING_CENTER': '#FF69B4'       # 热粉红
# }

# 新的标签映射
label_mapping = {
    'Continental arc': 'CA',
    'Island arc': 'IA',
    'Intra-oceanic arc': 'IOA',
    'BACK-ARC_BASIN': 'BAB',
    'SPREADING_CENTER': 'MOR',
    'OCEANIC PLATEAU': 'OP',
    'OCEAN ISLAND': 'OI',
    'CONTINENTAL FLOOD BASALT': 'CF',
    'CONTINENTAL_RIFT': 'CR'
}

# 对应的颜色列表
colors = [tectonic_colors.get(setting, 'gray') for setting in tectonic_settings]

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(26, 14))

# 读取本地世界地图图像
# 中文注释：底图是外部资产，按 README 放在 data/figures/assets 下。
world_map_path = str(WORLD_BASEMAP_PNG)
# 是否降低底图视觉强度：True 表示底图变浅，False 表示使用原始底图
reduce_basemap_intensity = True
# 底图变浅强度，数值越大越接近白色
basemap_fade_strength = 0.36
world_map = imread(world_map_path)

# 显示世界地图图像，可选择是否与白色混合以降低底图视觉强度，让样点更突出
if reduce_basemap_intensity:
    world_map_float = world_map.astype(np.float32)
    if world_map_float.max() > 1.0:   # 将 0-255 的图像归一化到 0-1
        world_map_float /= 255.0
    world_map_display = world_map_float * (1 - basemap_fade_strength) + basemap_fade_strength
else:
    world_map_display = world_map
ax.imshow(world_map_display, extent=[-180, 180, -90, 90], aspect='auto')

# 绘制数据点
# 中文注释：用轻微偏移的半透明黑点模拟阴影，让样点从底图上浮起一点。
shadow_longitudes = np.asarray(longitudes, dtype=float) + 0.55
shadow_latitudes = np.asarray(latitudes, dtype=float) - 0.35
ax.scatter(shadow_longitudes, shadow_latitudes, c='black', s=40, alpha=0.02, edgecolor='none', zorder=4)
# 中文注释：适当放大样点，让全球分布图中的点位更容易辨认。
sc = ax.scatter(longitudes, latitudes, c=colors, s=80, alpha=1, edgecolor='k', linewidth=0.7, zorder=5)

# 设置经纬度刻度和标签
ax.set_xticks(np.arange(-180, 181, 60))
ax.set_yticks(np.arange(-90, 91, 45))
ax.set_xticklabels(['180°W', '120°W', '60°W', '0°', '60°E', '120°E', '180°E'], fontsize=26)
ax.set_yticklabels(['90°S', '45°S', '0°', '45°N', '90°N'], fontsize=26)

ax.tick_params(axis='x', length=10, pad=20)  # 增加 x 轴标签的垂直间距
ax.tick_params(axis='y', length=10, pad=10)  # 增加 y 轴标签的水平间距

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

# 中文注释：显式设置整张地图面板外框，避免依赖 Matplotlib 默认边框样式。
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#000000')
    spine.set_linewidth(0.6)

# 添加图例
# 中文注释：固定图例顺序，避免受 tectonic_colors 字典定义顺序影响。
legend_order = [
    'Continental arc',
    'Island arc',
    'Intra-oceanic arc',
    'BACK-ARC_BASIN',
    'OCEANIC PLATEAU',
    'OCEAN ISLAND',
    'CONTINENTAL FLOOD BASALT',
    'SPREADING_CENTER',
    'CONTINENTAL_RIFT'
]
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                              label=f"{label_mapping[setting]}",
                              markerfacecolor=tectonic_colors[setting], markeredgecolor='k',
                              markeredgewidth=0.9, markersize=16)
                   for setting in legend_order]

# 将图例放在右下角
ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.79, -0.005),
          ncol=3, fontsize=24, columnspacing=0.8)

# 添加底图来源标注，避免图片使用时遗漏版权/来源信息
ax.text(0.936, 0.012, 'ESRI GEBCO Garmin',
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=16, color='#666666',
        # bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=3),
        zorder=10)
# 调整布局以适应图例
plt.tight_layout()
ax.set_aspect('auto')  # 或者使用 'equal'

# 保存图像
# 中文注释：输出正式现代玄武岩全球分布图。
output_path = str(MODERN_DISTRIBUTION_MAP_PNG)
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=200)

# 显示图像
# plt.show()
print('保存成功')
