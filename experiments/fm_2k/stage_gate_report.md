# TOPOTEX FM 2K Baseline Report

日期：2026-07-29 ｜ checkpoint：`checkpoints/baseline`（fm_2k，123,875 步 = 1982 mesh × 2000 exposures）
协议：`docs/experiment_protocol.md`（FM Euler-50，seed 20260727，shared Z_F）
本报告是 10K 扩展的决策门（三问：dataset quality / FM convergence / generalization）。

## 1. Dataset quality ✅

`dataset_statistics_final.json`：尝试 2000 → source 1987（成功率 **99.35%**，13 例失败全为
"reference render empty" 退化几何门禁）→ UV-query 1982；**零重复**（source + uv-query 双级），
full-query 覆盖率 **1.0**；faces 中位 1356；manifest SHA `05c696d1…` 全程入 checkpoint/record 可溯。

## 2. FM convergence ✅

- 训练：272 分钟（8×H800 DDP，packed K=4，bf16，243 mesh-exp/s），loss EMA 0.0519 平滑收敛，无 NaN/无回弹
- Stage-A 门禁（6.5% 预算处）已核查：无 collapse、无 UV shortcut、无 face 错位、partial 正常；
  worst8 全部为预算性欠拟合 → CONTINUE（`reports/fm_2k_stageA/`）
- Stage-A → 收官的指标增长：canonical 13.8→25.0、held-out 14.0→21.6、seam 比 4.4×→1.7×——
  欠拟合诊断被后续训练完全兑现

## 3. Generalization ✅

双 32-mesh 组（首 32 + offset 1000 不相交组）：

| 指标 | group0 | group1000 | fm_100 参考 |
|---|---|---|---|
| canonical / alternative | 25.00 / 24.91 | 25.52 / 25.67 | 23.6 / 23.2 |
| partial 区域 | 25.42 | 25.77 | 24.7 |
| **held-out UV family** | **21.56** | **21.63** | 20.4 / 18.5 |
| render 一致性 (GT fidelity) | 25.98 (26.11) | 26.72 (27.33) | 23.95 (24.57) |
| seam（GT 参照）| 0.11 (0.06) | 0.10 (0.06) | 0.10 (0.045) |

- **Z_F representation scaling 成立**：100→1982 mesh、架构零改动，held-out +1.2~3.1 dB，
  且两组数字高度一致（组间差 <0.6 dB ≪ 噪声底）
- 解耦签名保持（一致性与 GT fidelity 差 0.1~0.6 dB）
- seam 收敛到 GT 烘焙地板的 1.7-1.8×
- 局限：全部 1982 mesh 均参与训练——mesh 级泛化未测（held-out 轴是 UV family）；
  10K 阶段建议预留 unseen-mesh 验证集

## 效率档案

| 阶段 | 吞吐 | 备注 |
|---|---|---|
| 单卡 packed（基准）| 20.6 mesh-exp/s | benchmark_training.py |
| 8 卡 DDP 首版 | 21 | straggler + topo_pe 重算 |
| 8 卡 DDP 修复版 | **243** | PE 组缓存 + 同步尺寸类抽样 |

fm_2k 全程 4.5h（修复前估 49h）；单张纹理生成 0.22s（Euler-50）。

## 结论与 10K 建议

三项决策条件全部满足。**建议进入 10K 评估**，前置项：
1. 预留 unseen-mesh 验证集（如 9,500 训练 + 500 验证）补齐 mesh 级泛化轴
2. 源数据：eligible 池 22,490，需再建 ~8,000 source 样本（8 卡 ~17h，UniTEX 为主）
3. 训练预算：10K × 2000 exposures ≈ 23h（当前吞吐），必要时评估 exposures 递减调度
