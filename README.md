# SAM3

# pipeline
``` markdown
1 Step 1: Smart Resize
2 ━━━━━━━━━━━━━━━━━━
3 原图: 640 × 768 = 491,520 像素
4 min_pixels=200,704  max_pixels=1,003,520
5 491,520 在范围内 → 不需要缩放
6
7 但需要对齐到 patch_size×merge_size = 14×2 = 28 的倍数：
8   640 → 644 (最近的 28 的倍数: 28×23=644)
9   768 → 784 (28×28=784)
10 smart resize → 644 × 784
11
12 Step 2: Qwen ViT
13 ━━━━━━━━━━━━━━━━
14 patch_size=14:
15   644/14 = 46 patches (H)
16   784/14 = 56 patches (W)
17   → 46 × 56 = 2,576 tokens
18
19 spatial_merge_size=2 (merge 后给 LLM 的 token 数):
20   46/2 = 23
21   56/2 = 28
22   → 23 × 28 = 644 tokens 给 LLM
23
24 但 pre-merger grid (喂给 seg_head 的):
25   h_pre = 46, w_pre = 56
26
27 Step 3: Vision Proj + FPN
28 ━━━━━━━━━━━━━━━━━━━━━━━━
29 vision_proj: [2576, D_qwen] → [2576, 256] → reshape → [1, 256, 46, 56]
30
31 fpn_2x: [1, 256, 46, 56] → [1, 256, 92, 112]
32 fpn_4x: [1, 256, 46, 56] → [1, 256, 184, 224]
33
34 Step 4: MaskDecoder
35 ━━━━━━━━━━━━━━━━━━
36 TwoWayTransformer 在 46×56 = 2,576 tokens 上运行
37 output_upscaling 4x: [46×4, 56×4] = [184, 224]
38
39 high_res_features 融合:
40   feat_s1: conv_s1(92×112)  → [1, 64, 92, 112]
41   feat_s0: conv_s0(184×224) → [1, 32, 184, 224]
42
43 mask 输出: [Q, 1, 184, 224]
44
45 Step 5: Interpolate to Original
46 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
47 F.interpolate: [184, 224] → [640, 768]
48 放大倍数: ~3.5x (640/184) 和 ~3.4x (768/224)
49
50 Step 6: Loss
51 ━━━━━━━━━━━
52 pred mask: [640, 768]  vs  GT mask: [640, 768]
53 pixel-level Focal + Dice loss
```
# 新设计
<anchor\_1>到<anchor\_8>特殊token用于标记anchor，让模型学习到anchor特征；<target\_1>用于标记分割目标，提取目标hiddenstate用于分割。

$$loss = w_{text}*TextLoss+ w_{anchor}*AnchorCosineLoss + w_{seg}*TargetSegLoss(BCE+Dice+IoU)$$

# 在refcoco上进行Train和eval

## v1效果
![](./version1_example2.jpg)
![](./version1_example.jpg)


## v2效果
![](./version2_example1.jpg)
![](./version2_example2.jpg)
![](./version2_example3.jpg)

# anchor-target simple example
```json
{
    "image": "images/coco2017train/000000522418.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nSegment the sink that is to the bottom left of the doorknob."
      },
      {
        "from": "gpt",
        "reasoning": "I first find the doorknob<anchor_1>, then I look for the sink to its bottom left<target_1>.",
        "value": "It's sink<target_1>."
      }
    ],
    "masks": {
      "anchor": [
        {
          "token": "<anchor_1>",
          "id": "lvis:25333"
        }
      ],
      "target": [
        {
          "token": "<target_1>",
          "id": "lvis:25334"
        }
      ]
    }
  }
```
mask通过\[DatasetName:ID\]映射查询

这个样例数据过于简单，还需增强
