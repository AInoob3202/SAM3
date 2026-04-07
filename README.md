# SAM3

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
