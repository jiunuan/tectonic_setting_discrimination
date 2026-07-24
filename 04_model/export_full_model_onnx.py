# -*- coding: utf-8 -*-
"""
================================================================================
Full Model (ViT + Transformer 双流, 显式缺失编码 GeoDAN) -> ONNX 导出脚本
================================================================================
用途：把训练好的 PyTorch 权重导出成前端 onnxruntime-web 能加载的 model.onnx。

关键点（必须与 04_model/ablation_v4_vit_transformer.py 的 Full Model 完全一致）：
  · use_missing_mask = True
  · 矩阵分支输入 image:    [B, 2, 6, 6]  (通道0=分位数值/255, 通道1=缺失mask)
  · 序列分支输入 sequence: [B, 36, 2]    (特征0=值/255,      特征1=缺失mask)
  · 超参全部用默认 (embed_dim=128, num_heads=8, transformer_layers=2, ff_dim=256)
    —— 训练时仅传 num_classes 与 use_missing_mask，其余走默认。

模型结构在本文件内自包含复制，避免依赖 pandas/sklearn/matplotlib。
若改了 ablation_v4 的模型定义，这里要同步。

运行（babeldoc 环境已含 torch 2.5.1 + CUDA）：
  & "D:\\Program File\\anaconda\\envs\\babeldoc\\python.exe" \
      "E:\\program\\python\\basalt_tectonic_discrimination\\04_model\\export_full_model_onnx.py"
================================================================================
"""

import json
import os

import torch
import torch.nn as nn

# ── 路径配置 ────────────────────────────────────────────────────────────────
WEIGHTS_PATH = r"E:\program\python\basalt_tectonic_discrimination\data\models\Full_Model_(ViT+Transformer)_best_seed.pth"
# 直接写到前端 public/model/，导出后即可 npm run dev 验证
OUT_ONNX_PATH = r"E:\program\vue\tectonic_setting_discrimination_front_V2\public\model\model.onnx"
OUT_META_PATH = r"E:\program\vue\tectonic_setting_discrimination_front_V2\public\model\model_meta.json"

NUM_CLASSES = 9
USE_MISSING_MASK = True
OPSET = 17

# 类别顺序：由 pd.factorize(df_train['TECTONIC SETTING']) 的“首次出现序”决定，
# 取自训练文件 05_normalize_basalt_train.csv。下标 0..8 必须与前端 TECTONIC_SETTINGS 一致。
LABEL_ORDER = [
    "CONTINENTAL_RIFT",          # 0
    "OCEAN ISLAND",              # 1
    "SPREADING_CENTER",          # 2
    "Island arc",                # 3
    "CONTINENTAL FLOOD BASALT",  # 4
    "OCEANIC PLATEAU",           # 5
    "BACK-ARC_BASIN",            # 6
    "Intra-oceanic arc",         # 7
    "Continental arc",           # 8
]

# v1 列排布（与 ablation_v4 的 ORIGINAL_IMAGE_COLUMNS / COLUMNS_ELECTRODE_ORDER_V1 一致）
IMAGE_COLUMNS = [
    "NA2O(WT%)", "MGO(WT%)", "CR(PPM)", "AL2O3(WT%)", "SIO2(WT%)", "P2O5(WT%)",
    "K2O(WT%)", "CAO(WT%)", "TIO2(WT%)", "V(PPM)", "MNO(WT%)", "FEOT(WT%)",
    "RB(PPM)", "SR(PPM)", "Y(PPM)", "NB(PPM)", "CO(PPM)", "NI(PPM)",
    "BA(PPM)", "LA(PPM)", "CE(PPM)", "PR(PPM)", "ND(PPM)", "ZR(PPM)",
    "SM(PPM)", "EU(PPM)", "GD(PPM)", "TB(PPM)", "DY(PPM)", "HO(PPM)",
    "TH(PPM)", "ER(PPM)", "YB(PPM)", "LU(PPM)", "HF(PPM)", "TA(PPM)",
]
SEQ_COLUMNS = [
    "RB(PPM)", "K2O(WT%)", "BA(PPM)", "SR(PPM)", "CAO(WT%)", "NA2O(WT%)",
    "LA(PPM)", "Y(PPM)", "MGO(WT%)", "PR(PPM)", "CE(PPM)", "ER(PPM)",
    "HO(PPM)", "ND(PPM)", "SM(PPM)", "DY(PPM)", "LU(PPM)", "TB(PPM)",
    "GD(PPM)", "YB(PPM)", "EU(PPM)", "TH(PPM)", "AL2O3(WT%)", "HF(PPM)",
    "ZR(PPM)", "TIO2(WT%)", "MNO(WT%)", "V(PPM)", "NB(PPM)", "CR(PPM)",
    "TA(PPM)", "FEOT(WT%)", "CO(PPM)", "NI(PPM)", "SIO2(WT%)", "P2O5(WT%)",
]


# ── 模型定义（自 ablation_v4_vit_transformer.py 精确复制）────────────────────
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class ViT_Transformer_DualStream(nn.Module):
    def __init__(self, num_classes, input_size=6, patch_size=2,
                 embed_dim=128, num_heads=8, transformer_layers=2,
                 ff_dim=256, dropout=0.15, use_missing_mask=False):
        super().__init__()
        self.num_patches = (input_size // patch_size) ** 2   # 9
        self.seq_len = input_size * input_size               # 36
        self.embed_dim = embed_dim
        self.use_missing_mask = use_missing_mask
        image_channels = 2 if use_missing_mask else 1
        sequence_features = 2 if use_missing_mask else 1

        # 矩阵分支 (ViT)
        self.patch_embed = PatchEmbedding(image_channels, patch_size,
                                          embed_dim, self.num_patches)
        self.vit_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.vit_cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.vit_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])
        self.vit_norm = nn.LayerNorm(embed_dim)

        # 序列分支 (Transformer)
        self.seq_proj = nn.Linear(sequence_features, embed_dim)
        self.seq_norm = nn.LayerNorm(embed_dim)
        self.seq_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.seq_pos_embed = nn.Parameter(
            torch.randn(1, self.seq_len + 1, embed_dim) * 0.02)
        self.seq_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])
        self.seq_final_norm = nn.LayerNorm(embed_dim)

        # 分类头: 4 路特征 (vit_cls + vit_gap + seq_cls + seq_gap)
        head_in = embed_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(head_in, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, x, x_seq):
        batch_size = x.size(0)

        vit_tokens = self.patch_embed(x)                       # (B, 9, D)
        vit_cls = self.vit_cls.expand(batch_size, -1, -1)
        vit_cls = vit_cls + self.vit_cls_pos
        vit_tokens = torch.cat([vit_cls, vit_tokens], dim=1)   # (B, 10, D)
        for blk in self.vit_blocks:
            vit_tokens = blk(vit_tokens)
        vit_tokens = self.vit_norm(vit_tokens)
        vit_cls_out = vit_tokens[:, 0]
        vit_gap_out = vit_tokens[:, 1:].mean(dim=1)

        seq_tokens = self.seq_norm(self.seq_proj(x_seq))       # (B, 36, D)
        seq_cls = self.seq_cls.expand(batch_size, -1, -1)
        seq_tokens = torch.cat([seq_cls, seq_tokens], dim=1)   # (B, 37, D)
        seq_tokens = seq_tokens + self.seq_pos_embed
        for blk in self.seq_blocks:
            seq_tokens = blk(seq_tokens)
        seq_tokens = self.seq_final_norm(seq_tokens)
        seq_cls_out = seq_tokens[:, 0]
        seq_gap_out = seq_tokens[:, 1:].mean(dim=1)

        fused = torch.cat(
            [vit_cls_out, vit_gap_out, seq_cls_out, seq_gap_out], dim=1)
        return self.fusion(fused)


# ── 导出主流程 ──────────────────────────────────────────────────────────────
def main():
    device = torch.device("cpu")

    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(f"找不到权重文件: {WEIGHTS_PATH}")

    print(f"[加载权重] {WEIGHTS_PATH}")
    try:
        ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    except Exception:
        # 兼容含非张量对象的存档
        ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)

    state_dict = ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt and all(
        not torch.is_tensor(v) for k, v in ckpt.items() if k != "state_dict"
    ):
        state_dict = ckpt["state_dict"]

    model = ViT_Transformer_DualStream(
        num_classes=NUM_CLASSES, use_missing_mask=USE_MISSING_MASK).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[警告] 缺失参数键({len(missing)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"[警告] 多余参数键({len(unexpected)}): {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")
    if not missing and not unexpected:
        print("[OK] state_dict 完全匹配")
    model.eval()

    # dummy 输入：与前端张量形状一致
    dummy_img = torch.randn(4, 2, 6, 6, dtype=torch.float32)
    dummy_seq = torch.randn(4, 36, 2, dtype=torch.float32)

    with torch.no_grad():
        ref_out = model(dummy_img, dummy_seq)
    print(f"[前向自检] 输出 shape={tuple(ref_out.shape)} (应为 (4, {NUM_CLASSES}))")

    os.makedirs(os.path.dirname(OUT_ONNX_PATH), exist_ok=True)
    print(f"[导出 ONNX] -> {OUT_ONNX_PATH} (opset={OPSET})")
    torch.onnx.export(
        model,
        (dummy_img, dummy_seq),
        OUT_ONNX_PATH,
        input_names=["image", "sequence"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "sequence": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )
    print("[导出完成]")

    # 写出元信息，前端可读取它以避免硬编码类别顺序/列序
    meta = {
        "use_missing_mask": USE_MISSING_MASK,
        "num_classes": NUM_CLASSES,
        "label_order": LABEL_ORDER,
        "image_columns": IMAGE_COLUMNS,
        "seq_columns": SEQ_COLUMNS,
        "image_input": {"name": "image", "shape": [None, 2, 6, 6]},
        "seq_input": {"name": "sequence", "shape": [None, 36, 2]},
        "value_scale": 255.0,
        "notes": "通道0=分位数值/255, 通道1=缺失mask(1=缺失,0=实测); 现代数据KNN插补数值通道, 太古代数值通道缺失填0",
    }
    with open(OUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[写出元信息] -> {OUT_META_PATH}")

    # 可选：若装了 onnxruntime，做一次数值对齐校验
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(OUT_ONNX_PATH, providers=["CPUExecutionProvider"])
        ort_out = sess.run(
            ["logits"],
            {"image": dummy_img.numpy(), "sequence": dummy_seq.numpy()},
        )[0]
        max_diff = float(np.abs(ort_out - ref_out.numpy()).max())
        print(f"[ORT 校验] torch vs onnxruntime 最大绝对误差 = {max_diff:.3e}")
        if max_diff < 1e-3:
            print("[ORT 校验] 通过 [OK]")
        else:
            print("[ORT 校验] 误差偏大，请检查 [FAIL]")
    except ImportError:
        print("[提示] 未装 onnxruntime，跳过数值校验。"
              "如需校验: pip install onnxruntime")


if __name__ == "__main__":
    main()
