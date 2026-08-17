# Scenario-Wise-Rec / AdaptDHM 对 ADL 的实现审计

审计参照仓库提交 `60e63e596b3adc20a18bf9faddaf30f6f3c038c4`。Scenario-Wise-Rec 是多场景推荐 benchmark，统一提供多数据集、多模型和 CTR trainer；它不是 ADL 官方代码仓库。AdaptDHM 与 ADL 的核心轮廓非常接近：先从样本表征做动态聚类，再按 hard route 选择 cluster-specific MLP，并用共享权重和 cluster 权重的逐元素乘积表达参数。

## 符合论文核心思想的部分

1. 聚类输入来自样本 embedding 表征。
2. 用相似度、softmax assignment 和归一化中心做迭代更新。
3. 路由取 assignment argmax，模型按 cluster 选择参数。
4. cluster 参数以 shared weight 与 cluster weight 的 Hadamard product 表示，对应 ADL Eq. (6) 的主要结构。
5. center 更新不依赖监督标签，体现 ADL 自动发现隐式域的思想。

## 不能视为严格论文复刻的部分

1. AdaptDHM 把 centers 作为普通 tensor 属性，而不是 parameter/buffer。它不会自然进入 `state_dict`，也不会随标准 `.to(device)`、checkpoint 保存和恢复完整迁移；恢复模型可能丢失已经学到的动态域中心。
2. 代码在内部强制 `.to(self.device)`，设备状态依赖构造参数，而不是模块 buffer 机制，容易在加载或切换设备时不一致。
3. 原仓库实现中的 center/assignment 更新顺序没有用独立单元测试锁定；ADL Algorithm 1 对“3 次迭代后只做一次 EWMA”的顺序敏感。
4. Ali-CCP 运行脚本只由字段 `301` 构造 3 个域，而论文用场景指示、年龄、城市细分为 29 场景。三场景结果不能与论文 Table 1 直接比较。
5. 原脚本 AdaptDHM 使用 `[256,128,64,32,16,8]`，且输入特征选择与论文声明的统一五层 `[512,256,128,64,32]` 不同。
6. 原 AdaSparse 默认 epsilon=0.01、alpha=1.0，并未实现 AdaSparse 论文 Eq. (9)-(10) 的动态稀疏正则；因此它更像 Fusion gate 的简化版本。
7. 原训练脚本使用 batch=4096、weight decay、scheduler 等设置，与 ADL 论文公开的 Adam、batch=2048、lr=1e-3 口径不一致。
8. benchmark 读取的是 Ali-CCP sample CSV；不能据此验证论文报告的约 42.3M train / 43M test 完整数据结果。

## 本项目如何处理这些差异

- `DLMRouter.centers` 注册为 buffer，checkpoint、设备迁移和恢复均包含中心。
- `tests/test_router.py` 明确计算 Algorithm 1 期望值，锁定 3 次内部更新后一次 EWMA，并检查 eval 不更新中心。
- ADL 路由表征显式 detach，DLM 不参与反向传播。
- `paper29` 数据模式要求用户提供真实字段并强制 29 场景，不允许静默退化成三场景。
- SharedBottom、AdaSparse、ADL 默认统一五层 `[512,256,128,64,32]`。
- AdaSparse 使用实际 domain-aware 字段 embedding、hard Fusion gate、逐层稀疏诊断和动态稀疏辅助损失。
- 正式入口固定公开论文默认，并把所有差异参数写入 manifest。
- 选择 epoch 时跳过 test，之后 train+validation refit，再测试和多 seed 汇总。

结论：AdaptDHM 是理解和验证 ADL 结构思想的高价值近似实现，但不能作为 ADL 论文核心算法和实验协议的严谨官方复刻。本项目修正了模型状态、更新顺序、AdaSparse 正则和实验协议问题；仍无法替代论文未公开的 29 场景字段映射与完整数据预处理细节。
