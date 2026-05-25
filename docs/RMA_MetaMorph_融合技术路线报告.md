# RMA 与 MetaMorph 融合技术路线报告

## 1. 目标定义

你的目标可以表述为：

> 保留 MetaMorph “跨形态统一控制器”的主体框架，同时引入 RMA 中对整个物理环境与机器人本体的隐变量估计 `\hat{z}_t`，使策略在**不同形态**和**不同动力学/地形条件**下都具备在线自适应能力。

这件事本质上是在做一套 **Morphology-conditioned universal policy + online adaptation latent** 的混合架构。

我建议的总体方向不是把 RMA 整个 MLP 策略生硬塞进 MetaMorph，而是：

1. 保留 MetaMorph 的 per-limb transformer 主体。
2. 把 RMA 的 `\hat{z}_t` 改造成一个“历史驱动的 context latent”。
3. 让这个 latent 作为额外条件输入到 transformer 或 decoder。
4. 训练上采用“teacher privileged latent + student history estimator”的两阶段或三阶段方案。

这是最现实、风险最低、也最符合两个项目代码结构的路线。

---

## 2. 先说结论：推荐的融合方案

推荐采用 **“MetaMorph policy + RMA-style adaptation encoder”**：

- **Teacher 分支**：使用 privileged 信息构造真实环境上下文 `z_t`。
- **Student 分支**：使用一段历史观测 `h_t = (o_{t-H+1}, ..., o_t, a_{t-H+1}, ..., a_{t-1})` 预测 `\hat{z}_t`。
- **控制器主体**：仍使用 MetaMorph 的 transformer policy。
- **融合方式**：先从“late fusion 到每个 limb token 或 decoder”开始，不要一上来改 cross-attention。

最小可行版本建议：

`当前观测 tokens -> transformer -> [token features]`

并行增加：

`历史窗口 -> adaptation encoder -> \hat{z}_t`

然后把 `\hat{z}_t` 复制到每个 limb token 上做拼接，最后进入动作 decoder。

这是第一版最稳妥的方案。

---

## 3. 基于本地代码的关键观察

### 3.1 RMA 这边，当前仓库实际上是 “RMA/CMS 风格的蒸馏实现”

你本地的 `D:\CODES\master\rl_locomotion` 不是最原始的 RMA 官方仓库，而是基于 RMA 思路、面向复杂地形 locomotion 的版本。核心逻辑仍然保留了：

- privileged policy 先学习
- 再训练 history encoder 预测 latent
- 再让 blind/student policy 用 `\hat{z}_t` 控制

关键代码点：

- `raisimGymTorch/algo/ppo/module.py`
  - `MLPEncode`：privileged policy，显式把观测拆成 regular obs、prop latent、geom latent
  - `StateHistoryEncoder`：从历史观测估计 latent
- `raisimGymTorch/algo/ppo/dagger.py`
  - `DaggerExpert`：从 privileged obs 中提取 expert latent
  - `DaggerAgent.get_history_encoding()`：student 用 history 估计 latent
- `raisimGymTorch/env/envs/dagger_a1/Environment.hpp`
  - `observe()` 里把 `history_len * baseDim` 的历史和当前尾部特征拼起来

从当前配置看，latent 结构是比较“小而硬”的：

- `prop_latent_dim = 8`
- `geom_latent_dim = 1`
- 总 latent 大致是 `8 + (n_futures + 1) * 1`

也就是说，这个仓库里的 `\hat{z}_t` 不是一个超大表征，而是一个很紧凑的适应性 bottleneck。

### 3.2 MetaMorph 这边，主体是 per-limb token transformer

关键代码点：

- `metamorph/algos/ppo/model.py`
  - `TransformerModel`
  - `ActorCritic`
- `metamorph/envs/modules/agent.py`
  - `observation_step()`
  - `combine_limb_joint_obs()`
- `metamorph/envs/wrappers/multi_env_wrapper.py`
  - 做 limb padding / action padding

当前 MetaMorph 的输入特点：

- 主输入是 `proprioceptive`，按 limb 切成 token。
- token 中已经包含不少形态/硬件信息，比如：
  - `body_mass`
  - `body_shape`
  - `joint_range`
  - `joint_axis`
  - `gear`
- 环境外感知目前主要是 `hfield`，并且在 `model.py` 里只支持比较简单的外部特征融合。

一个很重要的现实点：

> 当前 `edges` 虽然被放进 observation 里，但 `TransformerModel.forward()` 实际并没有用图结构去约束注意力，主干仍然是标准 transformer encoder。

这意味着：

1. 你现在往 MetaMorph 里加 `\hat{z}_t`，工程上并不会和图结构强绑定。
2. 最容易落地的方式就是把 `\hat{z}_t` 当成一个全局 context，拼到 token 或 decoder 上。

---

## 4. 你真正要融合的，不是“RMA 的网络”，而是 “RMA 的训练范式”

这是最关键的一点。

RMA 的精髓不是某个 MLP 结构，而是这条链：

1. 用 privileged 信息学习一个有适应性的 teacher policy。
2. 把 privileged 环境与本体信息压缩成 latent `z_t`。
3. 让 student 仅靠历史观测去拟合 `\hat{z}_t`。
4. 控制器执行时不再访问 privileged 信息，而是用 `\hat{z}_t`。

所以和 MetaMorph 融合时，建议保留：

- MetaMorph 负责“跨形态共享控制器”
- RMA 负责“在线上下文自适应”

换句话说：

- **MetaMorph 回答：这是什么形态、每条 limb 应该如何协同？**
- **RMA 回答：当前地形/摩擦/质量扰动/执行器状态是什么？**

这两个信息源是互补的。

---

## 5. 推荐的融合架构

## 5.1 版本 A：最现实、最推荐

### 结构

1. **Per-limb current observation encoder**
   - 沿用 MetaMorph 当前的 `limb_embed`
2. **History adaptation encoder**
   - 新增一个 RMA-style `AdaptationEncoder`
   - 输入一段历史窗口
   - 输出全局 latent `\hat{z}_t \in R^{d_z}`
3. **Transformer trunk**
   - 仍对 limb tokens 做 self-attention
4. **Latent fusion**
   - 将 `\hat{z}_t` broadcast 到每个 limb
   - 与 transformer 输出 token feature 拼接
5. **Action decoder**
   - 每个 limb 输出对应动作

可写成：

`x_i^t = limb_embed(o_i^t)`

`\hat{z}_t = f_adapt(h_t)`

`u_i^t = Decoder(Transformer({x_i^t})_i concat \hat{z}_t)`

### 为什么推荐它

- 对 MetaMorph 改动最小。
- 不需要重写 transformer 层。
- 可以先验证 `\hat{z}_t` 是否真的提升泛化与适应。
- 训练和调参复杂度可控。

## 5.2 版本 B：把 `\hat{z}_t` 当成额外 token

做法：

- 在 limb tokens 前面插入一个 global adaptation token。
- token 内容来自 `\hat{z}_t` 的线性投影。
- transformer 通过注意力自动决定各 limb 如何使用它。

优点：

- 更“transformer-native”
- 信息传播自然

缺点：

- 训练更敏感
- 需要额外处理 padding / positional encoding / value head 的聚合逻辑

建议把它作为第二阶段增强版，而不是第一版。

## 5.3 版本 C：cross-attention 融合历史序列

做法：

- 不先把历史压成单个 `\hat{z}_t`
- 而是让当前 limb tokens 对 history tokens 做 cross-attention

这更强，但不建议一开始做，原因很现实：

- 改动大
- 显存贵
- 更难稳定
- 很难判断收益来自“历史建模”还是“更大模型”

---

## 6. `\hat{z}_t` 里应该放什么

如果你要忠实继承 RMA 思路，`z_t` 应该尽量覆盖“机器人自身 + 环境”的慢变量与隐变量，而不是仅仅塞一段 hfield。

建议把 teacher 可见的 privileged context 分成四类：

1. **形态/结构静态信息**
   - limb 数量
   - limb 类型
   - body shape
   - joint axis / range
   - gear
   - 几何拓扑统计量

2. **动力学参数**
   - body mass
   - 惯量或近似惯量描述
   - damping / armature
   - actuator strength
   - friction / restitution

3. **接触与地形信息**
   - hfield 局部窗口
   - 坡度/法向
   - 足端接触模式
   - slip 指标

4. **任务/运行上下文**
   - 目标速度
   - 外部扰动标记
   - 地形类型 ID（如果训练时可知）

但要注意：

> 不建议把 morphology static info 全塞进 `z_t`，因为 MetaMorph 当前 token 里已经显式编码了很多形态信息。

更合理的分工是：

- **形态显式信息**：继续放在 MetaMorph 的 token 输入中
- **时变上下文/隐变量**：交给 `\hat{z}_t`

所以第一版 `z_t` 更应该偏向：

- 质量扰动
- 摩擦变化
- 地形局部统计
- 执行器能力变化
- 接触模式和滑移特征

---

## 7. 具体到代码，应该怎么改

## 7.1 MetaMorph 侧新增 adaptation encoder

建议新增文件：

- `D:\CODES\master\metamorph\metamorph\algos\ppo\adaptation.py`

里面实现：

- `PrivilegedContextEncoder`
  - 输入 teacher 可见 privileged obs
  - 输出 `z_t`
- `HistoryAdaptationEncoder`
  - 输入历史窗口
  - 输出 `\hat{z}_t`

第一版可以直接借鉴 RMA 的 `StateHistoryEncoder` 思路：

- 先对每个时刻做线性投影
- 再做 1D temporal conv
- 最后输出固定维度 latent

这是比 LSTM 更稳、也更贴近你现有参考代码的做法。

## 7.2 修改 MetaMorph 的 observation 管线

你需要给 MetaMorph 增加两类观测：

1. **teacher privileged obs**
2. **student history buffer 的原始组成项**

建议改动位置：

- `metamorph/envs/modules/agent.py`
- `metamorph/envs/wrappers/`

建议新增字段：

- `privileged_context`
- `history_proprio`
- 可选：`history_action`

更现实的做法是先不要把整个 history 都放进 env observation dict，而是在 rollout storage 或 policy wrapper 里维护一个 history ring buffer。

原因：

- MetaMorph 当前 observation 体系是“单步观测”为主。
- 直接把 H 步历史塞进 env，会让 observation space 膨胀太多。
- 在 PPO buffer/policy 前向里维护历史更干净。

## 7.3 修改 `TransformerModel`

建议修改：

- `D:\CODES\master\metamorph\metamorph\algos\ppo\model.py`

新增逻辑：

1. policy forward 接收 `z_hat`
2. transformer 编码完 limb tokens 后
3. 将 `z_hat` broadcast 成 `(seq_len, batch, dz)`
4. 与 `obs_embed_t` 拼接
5. decoder 输入维度改为 `d_model + dz`

这是第一版最简单的融合点。

不要第一步就把 `z_hat` 融合到 `limb_embed` 前面，原因是：

- 当前观测 token 已经同时承担 morphology encoding 和 state encoding
- 太早融合会让表示纠缠，训练初期更不稳定
- 晚融合更容易做消融

## 7.4 修改 ActorCritic / PPO rollout

需要改：

- `metamorph/algos/ppo/model.py`
- `metamorph/algos/ppo/ppo.py`
- `metamorph/algos/ppo/buffer.py`（如果有 rollout buffer）

你需要支持两种前向模式：

1. **teacher mode**
   - 用 privileged encoder 得到 `z_t`
2. **student mode**
   - 用 history encoder 得到 `\hat{z}_t`

训练时至少要支持记录：

- 当前 obs
- 上一时刻动作
- history buffer
- privileged context
- `z_t`
- `\hat{z}_t`

---

## 8. 推荐训练流程

## 8.1 阶段 0：先做环境与观测对齐

这是最容易被低估的一步。

你要先确认：

1. MetaMorph 训练环境里是否能暴露足够的 privileged dynamics/terrain 信息
2. 这些信息是否在 reset 时和 step 时都能稳定获取
3. 多形态下 privileged context 的维度如何统一

这里的核心原则是：

> privileged context 必须是**跨形态统一维度**的。

建议做法：

- 静态 morphology 参数用统计量或固定槽位表达
- 地形信息直接用固定大小 hfield slice
- 动力学参数做 global vector

最终拼成固定维度 `privileged_context`。

## 8.2 阶段 1：训练 privileged teacher

第一阶段不要立即训练 student history encoder。

先训练：

- MetaMorph transformer policy
- 输入：当前单步观测 + privileged context latent `z_t`

实现方式有两种：

1. **直接把 privileged_context 输入 policy**
2. **先过 `PrivilegedContextEncoder` 压成 `z_t` 再输入 policy**

建议选第 2 种，因为这样后面 student 只需要拟合同一 latent 空间。

损失函数还是标准 PPO 为主。

## 8.3 阶段 2：固定 teacher，训练 adaptation encoder

这一步最接近 RMA。

冻结：

- transformer trunk
- action decoder
- privileged encoder（可选冻结）

训练：

- `HistoryAdaptationEncoder`

目标：

- `L_adapt = || \hat{z}_t - z_t ||^2`

可选增加：

- 动作蒸馏损失 `|| \pi(o_t, \hat{z}_t) - \pi(o_t, z_t) ||`
- value/distillation 辅助损失

我建议一开始就加一点动作蒸馏，不然仅做 latent regression，可能会出现“latent 数值接近但控制意义不对齐”的问题。

## 8.4 阶段 3：联合微调

当 student encoder 能稳定给出可用 `\hat{z}_t` 后，再做联合微调：

- transformer trunk
- decoder
- adaptation encoder

训练时逐步把 teacher latent `z_t` 替换为 student latent `\hat{z}_t`。

可以用 scheduled sampling 思路：

- 初期高概率用 `z_t`
- 中期混合
- 后期完全用 `\hat{z}_t`

这一步会明显提升真实闭环表现。

---

## 9. 一个可执行的最小实现顺序

如果你现在就开工，我建议按下面顺序做，不要跳步：

1. **只在 MetaMorph 中加入 privileged latent `z_t`**
   - 先不做 history encoder
   - 验证“上下文 latent 是否真的有收益”

2. **实现 history buffer + `HistoryAdaptationEncoder`**
   - 先只拟合 `z_t`

3. **把 policy 从 `z_t` 切换到 `\hat{z}_t`**
   - 做闭环验证

4. **再尝试更高级融合**
   - `z token`
   - cross-attention
   - per-limb adaptive latent

这个顺序非常重要，因为它能把问题拆开：

- 是 latent 定义有问题？
- 还是 history encoder 不行？
- 还是 fusion 点不对？

否则三件事一起改，最后很难定位失败原因。

---

## 10. 我最建议的 latent 定义

如果你问我第一版 `z_t` 该怎么设计，我会建议：

### 全局 latent `z_t`

维度先取：

- `d_z = 16` 或 `32`

teacher 输入由以下内容拼接：

- hfield 局部编码
- 全局动力学参数编码
  - mass statistics
  - damping / armature statistics
  - friction
  - actuator strength statistics
- 接触统计
  - 当前足端接触
  - 最近几步滑移/冲击摘要

而不要把所有 per-limb morphology 原始量都重新塞进去。

因为 MetaMorph 当前 token 已经有：

- `body_mass`
- `body_shape`
- `joint_range`
- `joint_axis`
- `gear`

如果你再在 `z_t` 里完整重复一遍，容易造成冗余和训练不稳定。

---

## 11. 可能遇到的三个大坑

## 11.1 坑一：形态变化和环境变化被混在一个 latent 里

如果 `z_t` 同时承担：

- morphology identity
- terrain context
- dynamics perturbation

那 student history encoder 很容易学不稳。

建议：

- morphology 尽量保持显式输入
- `z_t` 更聚焦时变上下文

## 11.2 坑二：历史窗口定义不合理

RMA 成功很依赖历史窗口设计。

对 MetaMorph，你至少要考虑：

- 过去 `H` 步 proprioceptive obs
- 过去 `H-1` 步 actions
- 可选：接触事件或外感知摘要

建议第一版：

- `H = 10` 或 `20`

不要一开始就上很长窗口，因为 PPO 训练和显存都会变重。

## 11.3 坑三：teacher 太强，student 拟合不到

如果 teacher 直接访问非常高维、非常精确的 privileged map，而 student 只能从短历史去猜，性能落差会很大。

所以 teacher 的 privileged context 也要适度“可蒸馏”：

- 用压缩后的 terrain/context 特征
- 不要把未来信息或过强 oracle 信息直接喂给 teacher

---

## 12. 我建议的实验设计

为了避免最后只得到一个“能跑但不知道为什么有效”的系统，建议至少做下面几组消融：

1. **MetaMorph baseline**
   - 不加 `z_t`

2. **MetaMorph + privileged `z_t`**
   - 验证 context latent 上限

3. **MetaMorph + history `\hat{z}_t`**
   - 验证 online adaptation 能否接近 teacher

4. **late fusion vs z-token**
   - 比较两种融合策略

5. **仅环境上下文 latent vs 环境+本体 latent**
   - 检查是否真的需要把 robot self 参数再编码进 `z_t`

评价指标建议包括：

- seen morphology 回报
- unseen morphology 回报
- dynamics shift 下回报
- terrain shift 下回报
- 恢复能力 / 扰动鲁棒性

---

## 13. 一个现实的开发排期

如果你是一个人推进，这个项目比较现实的节奏大概是：

### 第 1 周

- 跑通 MetaMorph baseline
- 明确 observation pipeline
- 定义 privileged context

### 第 2 周

- 加入 privileged encoder
- 训练 MetaMorph + `z_t` teacher

### 第 3 周

- 实现 history buffer
- 实现 adaptation encoder
- 做 latent regression

### 第 4 周

- 切换到 student `\hat{z}_t`
- 做联合微调和消融实验

如果中途环境改动较多，这个周期还会再拉长。

---

## 14. 最终建议

如果你的目标是“做出一个现实可行、能形成论文故事的系统”，我建议你采用下面这句非常清晰的路线：

> 用 MetaMorph 负责跨形态泛化，用 RMA 负责在线上下文适应；在实现上，将 RMA 的 history-based adaptation encoder 作为 MetaMorph transformer 的全局条件分支，并采用 privileged-teacher 到 history-student 的蒸馏训练范式。

这是当前最稳、最清楚、最容易做出有效结果的方案。

---

## 15. 我认为最值得你先做的第一步

不要一开始就改一大堆模块。

最值得先做的是：

1. 在 MetaMorph 里定义统一的 `privileged_context`
2. 加一个 `PrivilegedContextEncoder -> z_t`
3. 把 `z_t` late-fusion 到 transformer decoder
4. 先验证 teacher latent 是否提升性能

如果这一步没有明显收益，后面的 `\hat{z}_t` 学得再漂亮，意义也不会太大。

---

## 16. 附：与本地代码直接对应的改动入口

### RMA 参考入口

- `D:\CODES\master\rl_locomotion\raisimGymTorch\algo\ppo\module.py`
- `D:\CODES\master\rl_locomotion\raisimGymTorch\algo\ppo\dagger.py`
- `D:\CODES\master\rl_locomotion\raisimGymTorch\env\envs\dagger_a1\Environment.hpp`

### MetaMorph 主要改动入口

- `D:\CODES\master\metamorph\metamorph\algos\ppo\model.py`
- `D:\CODES\master\metamorph\metamorph\envs\modules\agent.py`
- `D:\CODES\master\metamorph\metamorph\envs\wrappers\multi_env_wrapper.py`
- `D:\CODES\master\metamorph\metamorph\envs\tasks\task.py`
- `D:\CODES\master\metamorph\metamorph\config.py`

---

## 17. 一句话总结

推荐你做的是：

**“在 MetaMorph 中加入一个 RMA 风格的 history-based adaptation latent `\hat{z}_t`，把它作为全局上下文条件去调制 transformer policy，而不是用它替代 MetaMorph 的 morphology tokenization。”**
