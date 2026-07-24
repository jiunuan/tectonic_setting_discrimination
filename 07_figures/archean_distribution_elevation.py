import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import imread
from pathlib import Path
import sys

# 中文注释：太古代预测全球分布图读取当前项目正式预测表和外部世界底图资产。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (
    ARCHEAN_FINAL_PREDICTIONS_CSV,
    ZENODO_ARCHEAN_PREDICTIONS_CSV,
    WORLD_BASEMAP_PNG,
    ARCHEAN_DISTRIBUTION_MAP_PNG,
)

# 中文注释：设置期刊图常用的无衬线字体和可编辑文本导出参数。
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "legend.frameon": True,
})

# 中文注释：读取太古代玄武岩 GeoDAN 预测结果。
file_path = str(ZENODO_ARCHEAN_PREDICTIONS_CSV if Path(ZENODO_ARCHEAN_PREDICTIONS_CSV).exists() else ARCHEAN_FINAL_PREDICTIONS_CSV)
df = pd.read_csv(file_path, encoding='utf-8-sig')

# 中文注释：提取经纬度和预测构造环境类别，并去掉缺少经纬度或预测类别的记录。
df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'pred_class_name']).copy()
df['LATITUDE'] = df['LATITUDE'].astype(float)
df['LONGITUDE'] = df['LONGITUDE'].astype(float)

# 中文注释：对完全重叠的经纬度做轻微、可复现的 jitter，避免多个样品互相遮挡。
rng = np.random.default_rng(20260626)
df['coord_count'] = df.groupby(['LATITUDE', 'LONGITUDE'])['LATITUDE'].transform('size')
df['plot_longitude'] = df['LONGITUDE']
df['plot_latitude'] = df['LATITUDE']
jitter_mask = df['coord_count'] > 1
df.loc[jitter_mask, 'plot_longitude'] = df.loc[jitter_mask, 'plot_longitude'] + rng.normal(0, 0.32, jitter_mask.sum())
df.loc[jitter_mask, 'plot_latitude'] = df.loc[jitter_mask, 'plot_latitude'] + rng.normal(0, 0.22, jitter_mask.sum())
df['plot_longitude'] = df['plot_longitude'].clip(-180, 180)
df['plot_latitude'] = df['plot_latitude'].clip(-89, 89)

# 中文注释：颜色方案与现代全球分布图保持一致，便于两张图直接对比。
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

# 中文注释：图例短标签与现代全球分布图保持一致。
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

# 中文注释：全图统一使用这个类别顺序；图例视觉顺序为 CA-IA-IOA / BAB-OP-OI / CF-MOR-CR。
category_order = [
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
legend_order = [
    'Continental arc', 'BACK-ARC_BASIN', 'CONTINENTAL FLOOD BASALT',
    'Island arc', 'OCEANIC PLATEAU', 'SPREADING_CENTER',
    'Intra-oceanic arc', 'OCEAN ISLAND', 'CONTINENTAL_RIFT'
]

# 中文注释：创建与现代全球分布图一致的宽幅地图画布。
fig, ax = plt.subplots(figsize=(26, 14))
ax.set_facecolor('white')

# 中文注释：读取与现代全球分布图相同的本地世界地图底图。
world_map_path = str(WORLD_BASEMAP_PNG)
world_map = imread(world_map_path)

# 中文注释：直接使用原始底图，不做淡化、去饱和或降对比处理。
# 中文注释：底图透明/淡化方式与现代全球分布图保持一致。
reduce_basemap_intensity = True
basemap_fade_strength = 0.36
if reduce_basemap_intensity:
    world_map_float = world_map.astype(np.float32)
    if world_map_float.max() > 1.0:   # 将 0-255 的图像归一化到 0-1
        world_map_float /= 255.0
    world_map_display = world_map_float * (1 - basemap_fade_strength) + basemap_fade_strength
else:
    world_map_display = world_map
ax.imshow(world_map_display, extent=[-180, 180, -90, 90], aspect='auto', zorder=1)

# 中文注释：点大小比现代图提高约 1.4 倍，白色 halo 和细黑边增强复杂底图上的识别度。
point_size = 185
halo_size = 220
for setting in category_order:
    setting_df = df[df['pred_class_name'] == setting]
    if setting_df.empty:
        continue
    ax.scatter(setting_df['plot_longitude'], setting_df['plot_latitude'],
               c='white', s=halo_size, alpha=0.82, edgecolor='none', zorder=4)
    ax.scatter(setting_df['plot_longitude'], setting_df['plot_latitude'],
               c=tectonic_colors[setting], s=point_size, alpha=1,
               edgecolor='black', linewidth=0.7, zorder=5)

# 中文注释：标注主要太古代克拉通区域，使用斜体和浅描边以减少对数据点的遮挡。
craton_labels = [
    # 中文注释：xy 指向数据/地质区域，xytext 放在更空的大陆或海洋位置，减少遮挡。
    {'name': 'Isua', 'xy': (-50.2, 65.2), 'xytext': (-45.0, 72.0), 'ha': 'left'},
    {'name': 'Abitibi', 'xy': (-77.8, 48.2), 'xytext': (-84.0, 36.0), 'ha': 'right'},
    {'name': 'Slave', 'xy': (-113.5, 65.0), 'xytext': (-134.0, 70.0), 'ha': 'right'},
    {'name': 'North China Craton', 'xy': (117.3, 38.5), 'xytext': (108.0, 42.0), 'ha': 'right'},
    {'name': 'Siberian', 'xy': (120.7, 57.0), 'xytext': (130.0, 61.0), 'ha': 'left'},
    {'name': 'Dharwar', 'xy': (77.3, 14.3), 'xytext': (72.0, 2.0), 'ha': 'right'},
    {'name': 'Singhbhum Craton', 'xy': (86.0, 22.0), 'xytext': (98.0, 14.0), 'ha': 'left'},
    {'name': 'Tanzania', 'xy': (32.4, -3.2), 'xytext': (18.0, 12.0), 'ha': 'right'},
    {'name': 'Kaapvaal', 'xy': (30.7, -26.1), 'xytext': (8.0, -36.0), 'ha': 'right'},
    {'name': 'Pilbara', 'xy': (119.3, -21.1), 'xytext': (136.0, -15.0), 'ha': 'left'},
    {'name': 'Yilgarn', 'xy': (121.5, -29.0), 'xytext': (139.0, -39.0), 'ha': 'left'}
]
for label in craton_labels:
    ax.annotate(label['name'], xy=label['xy'], xytext=label['xytext'],
                ha=label['ha'], va='center', fontsize=26, fontstyle='italic', fontweight='normal',
                color="#272626", zorder=8,
                arrowprops=dict(arrowstyle='-', color='#555555', linewidth=0.7,
                                alpha=0.65, shrinkA=2, shrinkB=4),
                path_effects=[pe.withStroke(linewidth=3.0, foreground='white', alpha=0.9)])

# 中文注释：设置经纬度刻度和标签，保留与现代全球分布图一致的视觉口径。
ax.set_xticks(np.arange(-180, 181, 60))
ax.set_yticks(np.arange(-90, 91, 45))
ax.set_xticklabels(['180°W', '120°W', '60°W', '0°', '60°E', '120°E', '180°E'], fontsize=26)
ax.set_yticklabels(['90°S', '45°S', '0°', '45°N', '90°N'], fontsize=26)
ax.tick_params(axis='x', length=10, pad=20, colors='black')
ax.tick_params(axis='y', length=10, pad=10, colors='black')

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

# 中文注释：Matplotlib 多列图例按列填充，因此这里用列优先顺序实现视觉上的 3×3 行优先布局。
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                              label=f"{label_mapping[setting]}",
                              markerfacecolor=tectonic_colors[setting], markeredgecolor='black',
                              markeredgewidth=0.8, markersize=17)
                   for setting in legend_order]
legend = ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.12, 0.40),
                   ncol=3, fontsize=24, columnspacing=0.4, handletextpad=0.35,
                   labelspacing=0.72, borderpad=0.35, framealpha=0.42)
legend.get_frame().set_facecolor((1, 1, 1, 0.42))
legend.get_frame().set_alpha(0.42)
legend.get_frame().set_edgecolor('none')

# 中文注释：添加底图来源标注，使用低对比灰色并贴近右下角。
ax.text(0.936, 0.012, 'ESRI GEBCO Garmin',
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=16, color='#666666',
        zorder=10)

# 中文注释：调整布局并保存太古代预测全球分布图。
plt.tight_layout()
ax.set_aspect('auto')

output_path = str(ARCHEAN_DISTRIBUTION_MAP_PNG)
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=250)

print('保存成功')
