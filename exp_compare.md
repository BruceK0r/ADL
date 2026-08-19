# 二域与三域推荐模型及 ADL 消融实验

本文比较 SharedBottom、不同 cluster 数量的 ADL，以及 Qwen、跨域历史和显式域
路由三项消融。实验分别在 Beauty + Electronic + Phone 三域数据，以及
Electronic + Phone 二域数据上独立完成。

> 评估口径为 sampled ranking：每条测试样本包含 1 个正物品和 100 个固定的
> 同域负物品，每个域 20,000 条测试上下文。结果不是全物品库检索指标。

## 1. 三域实验设置

### 1.1 固定设置

| 配置 | 数值 |
|---|---|
| 数据 | `data/processed/amazon_3domains` |
| 用户数 / 物品数 | 456,697 / 208,565 |
| Train / Validation / Test | 3,156,801 / 60,000 / 60,000 contexts |
| 候选数 | 训练 1+4；验证/测试 1+100 |
| 隐藏层 | 512 → 256 → 128 → 64 → 32 |
| Batch size | 2,048 contexts |
| Optimizer / LR | Adam / `1e-3` |
| 最大 epoch / Early stopping | 10 / patience=2 |
| 模型选择 | validation macro NDCG@10 |
| Seed | 2023 |
| ADL routing | 原始点积、3 次迭代、`beta=0.9` |

除被明确消融的部分外，数据划分、负样本、特征、训练参数和测试候选集合均
保持一致。K=3 使用此前同一数据、同一 seed 的正式 checkpoint，并重新评估以
补充域—簇统计；排序指标未改变。

### 1.2 比较方法

| 方法 | 改动与目的 |
|---|---|
| SharedBottom | 使用相同特征和五层 MLP，但所有样本共享一套普通线性层，不使用 DLM 或 cluster-specific 权重 |
| ADL K=1/3/5/7/9 | 只改变隐式 cluster 数量，考察 K 的影响 |
| ADL without Qwen | 不读取 Qwen 表示；候选、全局历史、同域历史改用可训练物品 ID embedding 投影到相同的 128 维空间 |
| ADL without cross-domain history | 保留候选 Qwen 和同域历史 Qwen，只将全局跨域历史分支置零 |
| ADL router with domain | 预测网络保持不变，仅把 8 维显式 domain embedding 加入 DLM 路由输入 |

`without Qwen` 没有简单删除历史，而是以物品 ID 表示替代文本表示，尽量保留
候选和序列信号。`without cross-domain history` 仍保留同域历史，因此比较的
是全局跨域历史提供的增量信息。

SharedBottom 与 ADL K=1 不等价：K=1 虽然只有一条固定路由，但每层仍使用
`W_shared ⊙ W_cluster` 的因子化权重。

## 2. 三域总体结果

| 方法 | Epoch | Val macro NDCG@10 | Test macro NDCG@10 | Domain-cluster NMI |
|---|---:|---:|---:|---:|
| SharedBottom | 2 | **0.15342** | 0.14664 | — |
| ADL K=1 | 1 | 0.14606 | **0.15222** | 0.0000 |
| ADL K=3 | 2 | 0.14350 | 0.13266 | 0.3445 |
| ADL K=5 | 2 | 0.14914 | 0.13919 | 0.4899 |
| ADL K=7 | 2 | 0.14793 | 0.13489 | 0.2665 |
| ADL K=9 | 1 | 0.14892 | 0.14215 | 0.4033 |
| ADL without Qwen | 5 | 0.10578 | 0.06697 | 0.0270 |
| ADL without cross-domain history | 2 | 0.14419 | 0.13813 | 0.0604 |
| ADL router with domain | 2 | 0.15127 | 0.14178 | 0.5103 |

NMI 衡量目标域与 hard route cluster 的相关程度：0 表示基本无关，1 表示两者
完全对应。SharedBottom 没有 router，因此不计算 NMI。

## 3. 三域结果分析

### 3.1 SharedBottom 与 ADL

- SharedBottom 获得最高 validation macro NDCG@10（`0.15342`）；ADL K=1
  获得最高 test macro NDCG@10（`0.15222`）。
- ADL K=1 的测试结果高于 SharedBottom `0.00558`，说明因子化权重可能有用；
  但 K=1 没有真正的动态分布划分。
- 所有 K>1 的 ADL 都没有超过 SharedBottom 或 K=1。因此，在当前单 seed、
  sampled-ranking 设置下，没有证据表明多 cluster 动态路由带来稳定收益。

### 3.2 cluster 数量 K

按 test macro NDCG@10 排序：

```text
K=1 (0.15222) > K=9 (0.14215) > K=5 (0.13919)
               > K=7 (0.13489) > K=3 (0.13266)
```

K 的效果明显非单调。增加 K 同时增加参数量，但没有稳定改善排序：

| K | 参数量（不含冻结 Qwen 表） | Test cluster counts |
|---:|---:|---|
| 1 | 22,243,288 | 6,060,000 |
| 3 | 23,066,648 | 1,193,300 / 1,975,598 / 2,891,102 |
| 5 | 23,890,008 | 1,132,657 / 479,173 / 1,606,683 / 269,744 / 2,571,743 |
| 7 | 24,713,368 | 2,052,945 / 1,917,844 / 22,862 / 1,216,405 / 601,709 / 116,770 / 131,465 |
| 9 | 25,536,728 | 5,086 / 41,538 / 553,400 / 2,178,530 / 473,111 / 716,960 / 1,151,286 / 1,709 / 938,380 |

K=7 和 K=9 都出现接近空簇的情况，例如 K=9 中两个簇仅接收 5,086 和
1,709 个候选，而总候选数为 6,060,000。这说明较大的 K 会造成明显的簇利用
不均衡；当前数据不支持“增加 K 就一定获得更细、更有效的分布划分”。

### 3.3 Qwen 文本特征贡献

移除 Qwen 后，test macro NDCG@10 从完整 K=3 的 `0.13266` 降至 `0.06697`，
绝对下降 `0.06569`，约下降 49.5%。三个域均明显下降：

| 域 | 完整 ADL K=3 | Without Qwen | 差值 |
|---|---:|---:|---:|
| Beauty | 0.15170 | 0.07773 | -0.07397 |
| Electronic | 0.14743 | 0.07906 | -0.06837 |
| Phone | 0.09885 | 0.04413 | -0.05472 |

这表明 Qwen 内容表示是当前系统最重要的增量特征之一。域内物品很多，纯 ID
embedding 对长尾、冷物品以及历史语义相似性的表达明显不足。

### 3.4 跨域历史贡献

移除全局跨域历史后，test macro NDCG@10 反而从 `0.13266` 上升至
`0.13813`。Beauty、Electronic、Phone 的 NDCG@10 分别提高 `0.00252`、
`0.00260`、`0.01130`。

在这个受控消融中，数据和测试候选完全相同，因此结果提示当前“所有域历史
Qwen 向量简单求均值”的做法可能引入噪声或负迁移，尤其对 Phone 更明显。
不过差异仍较小且只有一个 seed，不能据此断言跨域行为本身无用；更合理的
下一步是使用注意力、时间衰减或目标域条件化聚合，而不是简单平均。

### 3.5 router 是否应加入显式 domain

| Router | Test macro NDCG@10 | Domain-cluster NMI |
|---|---:|---:|
| 不含 domain（ADL K=3） | 0.13266 | 0.3445 |
| 含 domain embedding | 0.14178 | 0.5103 |

加入 domain 后排序分数提高 `0.00912`，同时 NMI 从 `0.3445` 升至 `0.5103`。
最佳 checkpoint 的各域 cluster 比例如下：

| Router | 域 | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---:|---:|---:|
| 不含 domain | Beauty | 19.18% | 80.63% | 0.19% |
| 不含 domain | Electronic | 26.17% | 5.01% | 68.83% |
| 不含 domain | Phone | 13.73% | 12.17% | 74.10% |
| 含 domain | Beauty | 0.15% | 7.32% | 92.52% |
| 含 domain | Electronic | 67.00% | 32.96% | 0.05% |
| 含 domain | Phone | 74.45% | 25.04% | 0.51% |

结论是：即使 router 不显式读取 domain，它也会从物品内容和历史中学到明显的
域结构，主要把 Beauty 与 Electronic/Phone 分开；加入 domain 后这种分界变得
更强。但 cluster 并不是三个域的一一映射，Electronic 与 Phone 仍共享近似的
cluster 混合。

如果目标是提高当前三域排序效果，router 加入 domain 在本次实验中更好；如果
目标是让 ADL 发现超越人工域标签的隐式分布，则不应加入 domain，并应继续
降低内容中的显式域捷径和检查 cluster 稳定性。

### 3.6 结论与限制

本次单 seed 结果支持以下判断：

1. Qwen 文本/内容表示贡献最大，不能轻易移除；
2. 当前简单的全局跨域历史均值没有带来收益，可能存在负迁移；
3. K 的选择敏感且非单调，较大 K 会产生近空簇；
4. 显式 domain 能提高排序并增强域—簇对应，但会让 router 更接近域分类器；
5. 当前最稳妥的基线仍是 SharedBottom 和 ADL K=1，而不是多 cluster ADL。

这些结论仍受单随机种子、固定负样本和 sampled-ranking 协议限制。正式论文结论
建议至少补充 3–5 个 seeds，并报告均值、标准差和 cluster 稳定性。

## 4. 三域完整逐域指标

### 4.1 Recall

| 方法 | 域 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---|---:|---:|---:|---:|---:|
| SharedBottom | Beauty | 0.05915 | 0.19505 | 0.29525 | 0.44685 | 0.80780 |
| SharedBottom | Electronic | 0.05325 | 0.19275 | 0.31475 | 0.50320 | 0.82370 |
| SharedBottom | Phone | 0.02385 | 0.13310 | 0.26215 | 0.49750 | 0.86835 |
| ADL K=1 | Beauty | 0.05490 | 0.20440 | 0.31765 | 0.48170 | 0.80165 |
| ADL K=1 | Electronic | 0.05435 | 0.20330 | 0.33275 | 0.51995 | 0.82605 |
| ADL K=1 | Phone | 0.02365 | 0.13425 | 0.26715 | 0.49720 | 0.84740 |
| ADL K=3 | Beauty | 0.04860 | 0.18630 | 0.29195 | 0.45640 | 0.82245 |
| ADL K=3 | Electronic | 0.04345 | 0.17400 | 0.29650 | 0.48230 | 0.81330 |
| ADL K=3 | Phone | 0.02070 | 0.10620 | 0.22225 | 0.46065 | 0.84910 |
| ADL K=5 | Beauty | 0.04750 | 0.18010 | 0.27985 | 0.42470 | 0.79410 |
| ADL K=5 | Electronic | 0.04965 | 0.18955 | 0.31240 | 0.49995 | 0.81725 |
| ADL K=5 | Phone | 0.02475 | 0.12565 | 0.24830 | 0.47615 | 0.85485 |
| ADL K=7 | Beauty | 0.05140 | 0.18125 | 0.28045 | 0.41870 | 0.77565 |
| ADL K=7 | Electronic | 0.04925 | 0.17740 | 0.29645 | 0.48210 | 0.81755 |
| ADL K=7 | Phone | 0.02270 | 0.11655 | 0.22980 | 0.44445 | 0.85235 |
| ADL K=9 | Beauty | 0.05405 | 0.18885 | 0.29005 | 0.43050 | 0.75830 |
| ADL K=9 | Electronic | 0.05070 | 0.19455 | 0.31680 | 0.50560 | 0.82605 |
| ADL K=9 | Phone | 0.02430 | 0.11975 | 0.24310 | 0.47245 | 0.84960 |
| ADL without Qwen | Beauty | 0.02490 | 0.09310 | 0.15160 | 0.24905 | 0.79395 |
| ADL without Qwen | Electronic | 0.02305 | 0.09630 | 0.15750 | 0.28565 | 0.75515 |
| ADL without Qwen | Phone | 0.01180 | 0.05110 | 0.09105 | 0.21965 | 0.85090 |
| ADL without cross-domain history | Beauty | 0.05405 | 0.18805 | 0.29000 | 0.43515 | 0.79345 |
| ADL without cross-domain history | Electronic | 0.04520 | 0.17545 | 0.29925 | 0.48290 | 0.80805 |
| ADL without cross-domain history | Phone | 0.02320 | 0.11905 | 0.24635 | 0.47475 | 0.83830 |
| ADL router with domain | Beauty | 0.05120 | 0.19255 | 0.30390 | 0.46350 | 0.82365 |
| ADL router with domain | Electronic | 0.05025 | 0.18385 | 0.30830 | 0.50155 | 0.81955 |
| ADL router with domain | Phone | 0.02455 | 0.11805 | 0.24690 | 0.47630 | 0.86265 |

### 4.2 NDCG

| 方法 | 域 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---|---:|---:|---:|---:|---:|
| SharedBottom | Beauty | 0.05915 | 0.12757 | 0.15972 | 0.19771 | 0.26890 |
| SharedBottom | Electronic | 0.05325 | 0.12301 | 0.16210 | 0.20942 | 0.27310 |
| SharedBottom | Phone | 0.02385 | 0.07674 | 0.11811 | 0.17696 | 0.25101 |
| ADL K=1 | Beauty | 0.05490 | 0.13009 | 0.16637 | 0.20760 | 0.27078 |
| ADL K=1 | Electronic | 0.05435 | 0.12861 | 0.17009 | 0.21725 | 0.27810 |
| ADL K=1 | Phone | 0.02365 | 0.07767 | 0.12020 | 0.17792 | 0.24765 |
| ADL K=3 | Beauty | 0.04860 | 0.11772 | 0.15170 | 0.19294 | 0.26518 |
| ADL K=3 | Electronic | 0.04345 | 0.10814 | 0.14743 | 0.19403 | 0.25972 |
| ADL K=3 | Phone | 0.02070 | 0.06191 | 0.09885 | 0.15851 | 0.23593 |
| ADL K=5 | Beauty | 0.04750 | 0.11363 | 0.14568 | 0.18211 | 0.25452 |
| ADL K=5 | Electronic | 0.04965 | 0.11955 | 0.15904 | 0.20617 | 0.26928 |
| ADL K=5 | Phone | 0.02475 | 0.07368 | 0.11285 | 0.17003 | 0.24562 |
| ADL K=7 | Beauty | 0.05140 | 0.11690 | 0.14879 | 0.18337 | 0.25324 |
| ADL K=7 | Electronic | 0.04925 | 0.11305 | 0.15124 | 0.19781 | 0.26447 |
| ADL K=7 | Phone | 0.02270 | 0.06848 | 0.10465 | 0.15830 | 0.23959 |
| ADL K=9 | Beauty | 0.05405 | 0.12163 | 0.15419 | 0.18949 | 0.25361 |
| ADL K=9 | Electronic | 0.05070 | 0.12253 | 0.16178 | 0.20911 | 0.27275 |
| ADL K=9 | Phone | 0.02430 | 0.07118 | 0.11049 | 0.16806 | 0.24345 |
| ADL without Qwen | Beauty | 0.02490 | 0.05890 | 0.07773 | 0.10213 | 0.20831 |
| ADL without Qwen | Electronic | 0.02305 | 0.05941 | 0.07906 | 0.11086 | 0.20450 |
| ADL without Qwen | Phone | 0.01180 | 0.03134 | 0.04413 | 0.07559 | 0.20233 |
| ADL without cross-domain history | Beauty | 0.05405 | 0.12138 | 0.15422 | 0.19063 | 0.26086 |
| ADL without cross-domain history | Electronic | 0.04520 | 0.11033 | 0.15003 | 0.19619 | 0.26079 |
| ADL without cross-domain history | Phone | 0.02320 | 0.06956 | 0.11015 | 0.16744 | 0.24000 |
| ADL router with domain | Beauty | 0.05120 | 0.12194 | 0.15779 | 0.19782 | 0.26903 |
| ADL router with domain | Electronic | 0.05025 | 0.11667 | 0.15652 | 0.20504 | 0.26836 |
| ADL router with domain | Phone | 0.02455 | 0.06998 | 0.11103 | 0.16856 | 0.24567 |

## 5. 三域结果与复现位置

- 实验根目录：`outputs/compare_3domains_seed2023/`
- 汇总 JSON：`outputs/compare_3domains_seed2023/comparison.json`
- 汇总 CSV：`outputs/compare_3domains_seed2023/comparison.csv`
- 各实验完整结果：`outputs/compare_3domains_seed2023/<experiment>/results.json`
- 训练日志：`outputs/compare_3domains_seed2023/logs/`
- 训练入口：`scripts/run_3domain_experiment.py`
- 汇总入口：`scripts/summarize_3domain_compare.py`

## 6. 二域实验设置

二域实验只使用 Electronic + Phone，模型实现、评估代码和训练超参数均与三域
实验一致；除域集合及由此重新生成的数据外，不改变其他实验条件。

| 配置 | 数值 |
|---|---|
| 数据 | `data/processed/amazon_ele_phone` |
| 用户数 / 物品数 | 290,876 / 116,966 |
| Electronic / Phone 物品数 | 96,642 / 20,324 |
| Train / Validation / Test | 1,777,346 / 40,000 / 40,000 contexts |
| 每域 Validation / Test | 20,000 / 20,000 contexts |
| 候选数 | 训练 1+4；验证/测试 1+100 |
| 模型、优化器与 early stopping | 与三域实验相同 |
| 模型选择 | validation macro NDCG@10 |
| Seed | 2023 |

ADL K=3 复用此前在同一二域数据、同一 seed 和同一训练设置下得到的正式
checkpoint，并重新评估以补齐域—簇统计；其余八组均在本轮重新训练。

## 7. 二域总体结果

| 方法 | Epoch | Val macro NDCG@10 | Test macro NDCG@10 | Domain-cluster NMI |
|---|---:|---:|---:|---:|
| SharedBottom | 1 | **0.16297** | 0.15997 | — |
| ADL K=1 | 1 | 0.15094 | 0.15167 | 0.0000 |
| ADL K=3 | 1 | 0.15857 | **0.16294** | 0.0342 |
| ADL K=5 | 2 | 0.15006 | 0.14179 | 0.0279 |
| ADL K=7 | 1 | 0.15043 | 0.14611 | 0.0195 |
| ADL K=9 | 2 | 0.15581 | 0.15994 | 0.1838 |
| ADL without Qwen | 3 | 0.10437 | 0.06089 | 0.0558 |
| ADL without cross-domain history | 2 | 0.14305 | 0.13214 | 0.0112 |
| ADL router with domain | 1 | 0.15442 | 0.15799 | 0.3209 |

## 8. 二域结果分析

### 8.1 SharedBottom 与 cluster 数量

二域测试集上，ADL K=3 的 macro NDCG@10 最高（`0.16294`），比
SharedBottom 的 `0.15997` 高 `0.00297`；ADL K=9 的 `0.15994` 与
SharedBottom 基本持平。不同 K 的结果为：

```text
K=3 (0.16294) > K=9 (0.15994) > K=1 (0.15167)
                > K=7 (0.14611) > K=5 (0.14179)
```

K 的影响仍然非单调，不能由“二域”直接推出最合适的 K=2，也不能认为增加
cluster 数一定提高效果。较大 K 同样出现利用不均衡：

| K | 参数量（不含冻结 Qwen 表） | Test cluster counts |
|---:|---:|---|
| 1 | 14,005,840 | 4,040,000 |
| 3 | 14,829,200 | 1,179,695 / 2,762,788 / 97,517 |
| 5 | 15,652,560 | 999,141 / 747,718 / 5,440 / 24,363 / 2,263,338 |
| 7 | 16,475,920 | 1,052,106 / 418,870 / 181,016 / 2,357,691 / 4,193 / 6,506 / 19,618 |
| 9 | 17,299,280 | 1,299,621 / 98,883 / 663,519 / 13,265 / 23,135 / 163,638 / 112,334 / 1,661,468 / 4,137 |

K=5、7、9 都有接近空簇，表明额外参数没有转化为稳定、均衡的隐式分布。
另外，SharedBottom 的 validation 分数最高，而 K=3 的 test 分数最高；单 seed
下这类小差异不宜解释为确定优势。

### 8.2 Qwen 文本特征贡献

移除 Qwen 后，test macro NDCG@10 从完整 K=3 的 `0.16294` 降至
`0.06089`，绝对下降 `0.10205`，约下降 62.6%。

| 域 | 完整 ADL K=3 | Without Qwen | 差值 |
|---|---:|---:|---:|
| Electronic | 0.18053 | 0.08105 | -0.09948 |
| Phone | 0.14536 | 0.04073 | -0.10462 |

两个域都严重下降，Phone 的相对降幅更大。与三域实验一致，Qwen 内容表示是
当前模型最关键的特征之一，纯物品 ID embedding 无法替代其语义与长尾泛化能力。

### 8.3 跨域历史贡献

移除全局跨域历史后，test macro NDCG@10 从 `0.16294` 降至 `0.13214`，
绝对下降 `0.03080`，约下降 18.9%。Electronic 和 Phone 分别下降
`0.02826` 与 `0.03334`。

这说明 Electronic + Phone 二域之间的跨域历史有明确增益。它与三域实验中
“移除跨域历史略有提升”的现象相反：Electronic 与 Phone 的语义和消费场景
更接近，而加入 Beauty 后，简单平均全部域历史可能混入异质行为并造成负迁移。
因此本实验支持的是“跨域历史是否有用取决于域组合和聚合方式”，而不是一个
对所有域集合都成立的结论。

### 8.4 router 是否应加入显式 domain

| Router | Test macro NDCG@10 | Domain-cluster NMI |
|---|---:|---:|
| 不含 domain（ADL K=3） | **0.16294** | 0.0342 |
| 含 domain embedding | 0.15799 | **0.3209** |

显式加入 domain 后，NMI 明显提高，但排序分数下降 `0.00495`。各域 hard route
比例如下：

| Router | 域 | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---:|---:|---:|
| 不含 domain | Electronic | 33.84% | 61.34% | 4.83% |
| 不含 domain | Phone | 24.56% | 75.43% | <0.01% |
| 含 domain | Electronic | 45.51% | 26.77% | 27.71% |
| 含 domain | Phone | 99.80% | 0.01% | 0.18% |

显式 domain 几乎把全部 Phone 候选压到 Cluster 0，使 router 更像域分类器。
Electronic 的 NDCG@10 略升 `0.00277`，但 Phone 下降 `0.01268`，最终 macro
指标变差。因此更高的域—簇对应关系不等于更好的推荐效果；在当前二域设置中，
不应仅为了提高 cluster 可解释性就给 router 加入 domain。

## 9. 二域完整逐域指标

### 9.1 Recall

| 方法 | 域 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---|---:|---:|---:|---:|---:|
| SharedBottom | Electronic | 0.06625 | 0.22635 | 0.35535 | 0.54555 | 0.84860 |
| SharedBottom | Phone | 0.03095 | 0.14375 | 0.28780 | 0.51560 | 0.86465 |
| ADL K=1 | Electronic | 0.05710 | 0.21360 | 0.33605 | 0.52350 | 0.83290 |
| ADL K=1 | Phone | 0.03015 | 0.14285 | 0.27825 | 0.50230 | 0.85030 |
| ADL K=3 | Electronic | 0.05920 | 0.21520 | 0.34935 | 0.53685 | 0.83290 |
| ADL K=3 | Phone | 0.03240 | 0.16235 | 0.31665 | 0.53605 | 0.85840 |
| ADL K=5 | Electronic | 0.04745 | 0.18705 | 0.31700 | 0.51320 | 0.82470 |
| ADL K=5 | Phone | 0.02505 | 0.13570 | 0.28295 | 0.51605 | 0.85765 |
| ADL K=7 | Electronic | 0.05025 | 0.19655 | 0.32020 | 0.50550 | 0.81705 |
| ADL K=7 | Phone | 0.02840 | 0.14455 | 0.28250 | 0.49285 | 0.83850 |
| ADL K=9 | Electronic | 0.04960 | 0.19790 | 0.32940 | 0.52105 | 0.82520 |
| ADL K=9 | Phone | 0.03775 | 0.17955 | 0.32495 | 0.53585 | 0.85800 |
| ADL without Qwen | Electronic | 0.02235 | 0.09535 | 0.16660 | 0.28195 | 0.75110 |
| ADL without Qwen | Phone | 0.01085 | 0.04755 | 0.08540 | 0.17355 | 0.77960 |
| ADL without cross-domain history | Electronic | 0.04545 | 0.17780 | 0.30660 | 0.49995 | 0.81845 |
| ADL without cross-domain history | Phone | 0.02295 | 0.12025 | 0.25210 | 0.48155 | 0.84110 |
| ADL router with domain | Electronic | 0.06010 | 0.21780 | 0.35525 | 0.53815 | 0.83530 |
| ADL router with domain | Phone | 0.03165 | 0.15030 | 0.28560 | 0.50735 | 0.84880 |

### 9.2 NDCG

| 方法 | 域 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---|---:|---:|---:|---:|---:|
| SharedBottom | Electronic | 0.06625 | 0.14684 | 0.18819 | 0.23605 | 0.29640 |
| SharedBottom | Phone | 0.03095 | 0.08582 | 0.13176 | 0.18912 | 0.25871 |
| ADL K=1 | Electronic | 0.05710 | 0.13552 | 0.17476 | 0.22191 | 0.28349 |
| ADL K=1 | Phone | 0.03015 | 0.08526 | 0.12857 | 0.18486 | 0.25438 |
| ADL K=3 | Electronic | 0.05920 | 0.13747 | 0.18053 | 0.22773 | 0.28651 |
| ADL K=3 | Phone | 0.03240 | 0.09589 | 0.14536 | 0.20052 | 0.26473 |
| ADL K=5 | Electronic | 0.04745 | 0.11664 | 0.15836 | 0.20769 | 0.26966 |
| ADL K=5 | Phone | 0.02505 | 0.07831 | 0.12521 | 0.18372 | 0.25191 |
| ADL K=7 | Electronic | 0.05025 | 0.12342 | 0.16305 | 0.20965 | 0.27154 |
| ADL K=7 | Phone | 0.02840 | 0.08505 | 0.12918 | 0.18205 | 0.25093 |
| ADL K=9 | Electronic | 0.04960 | 0.12348 | 0.16569 | 0.21392 | 0.27437 |
| ADL K=9 | Phone | 0.03775 | 0.10760 | 0.15418 | 0.20723 | 0.27139 |
| ADL without Qwen | Electronic | 0.02235 | 0.05818 | 0.08105 | 0.10986 | 0.20176 |
| ADL without Qwen | Phone | 0.01085 | 0.02866 | 0.04073 | 0.06271 | 0.17818 |
| ADL without cross-domain history | Electronic | 0.04545 | 0.11094 | 0.15227 | 0.20083 | 0.26415 |
| ADL without cross-domain history | Phone | 0.02295 | 0.06998 | 0.11202 | 0.16957 | 0.24118 |
| ADL router with domain | Electronic | 0.06010 | 0.13922 | 0.18330 | 0.22931 | 0.28840 |
| ADL router with domain | Phone | 0.03165 | 0.08951 | 0.13268 | 0.18850 | 0.25665 |

## 10. 二域与三域对照结论

| 观察项 | 三域 | 二域 | 结论 |
|---|---:|---:|---|
| 最佳 Test macro NDCG@10 | ADL K=1，0.15222 | ADL K=3，0.16294 | 最优 K 随域组合变化 |
| Without Qwen 相对完整 K=3 | -0.06569 | -0.10205 | 两种设置都强依赖文本特征 |
| Without cross-domain history 相对完整 K=3 | +0.00547 | -0.03080 | 简单跨域历史在相近域有益，在异质域组合中可能负迁移 |
| Router with domain 相对完整 K=3 | +0.00912 | -0.00495 | 域—簇对齐增强不保证排序收益 |

二域与三域数据是独立预处理、独立训练得到的实验集合，用户/物品规模和时间
切分阈值不同，因此不应把两个绝对分数直接当作“减少一个域带来的提升”。更
可靠的比较是观察各自内部的受控消融方向。总体上，Qwen 的贡献最稳定；K、
跨域历史和显式 domain 的收益都依赖具体域组合。

二域实验的结果与复现位置：

- 实验根目录：`outputs/compare_ele_phone_seed2023/`
- 汇总 JSON：`outputs/compare_ele_phone_seed2023/comparison.json`
- 汇总 CSV：`outputs/compare_ele_phone_seed2023/comparison.csv`
- 各实验完整结果：`outputs/compare_ele_phone_seed2023/<experiment>/results.json`
- 训练日志：`outputs/compare_ele_phone_seed2023/logs/`
