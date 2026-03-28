# 面向强化学习的谱感知矩阵优化器：从 Muon 到 Schatten-p 最速下降与矩阵自然梯度

> 技术报告 · 2026-03-28
> 
> 本文从 Muon 优化器的范数-约束优化理论出发，结合 RLVR 训练动力学的最新发现（Three-Gate Theory），分析 Muon 在 RL 场景中的结构性缺陷，并推导两种 RL-native 的矩阵优化器：(1) 基于 Schatten-p 范数的最速下降方法；(2) 基于矩阵自然梯度的逆谱变换方法。全部推导从第一性原理出发，不跳步骤。

-----

## 目录

- [1. 预备知识](#1-预备知识)
  - [1.1 矩阵范数与奇异值分解](#11-矩阵范数与奇异值分解)
  - [1.2 范数视角下的最速下降](#12-范数视角下的最速下降)
- [2. Muon 优化器的数学原理](#2-muon-优化器的数学原理)
  - [2.1 从向量到矩阵：Sign 到 Msign](#21-从向量到矩阵sign-到-msign)
  - [2.2 约束优化视角](#22-约束优化视角)
  - [2.3 特征空间的对偶解释](#23-特征空间的对偶解释)
- [3. RLVR 的优化动力学](#3-rlvr-的优化动力学)
  - [3.1 参数更新稀疏性](#31-参数更新稀疏性)
  - [3.2 Three-Gate Theory](#32-three-gate-theory)
  - [3.3 Off-Principal 方向学习](#33-off-principal-方向学习)
- [4. Muon 在 RL 中的结构性缺陷](#4-muon-在-rl-中的结构性缺陷)
  - [4.1 各向同性更新与 Off-Principal 偏好的冲突](#41-各向同性更新与-off-principal-偏好的冲突)
  - [4.2 谱结构保护的失败](#42-谱结构保护的失败)
  - [4.3 Hopper 的经验修正及其局限](#43-hopper-的经验修正及其局限)
- [5. 方法一：Schatten-p 范数下的最速下降](#5-方法一schatten-p-范数下的最速下降)
  - [5.1 向量情形：Lp 范数最速下降](#51-向量情形lp-范数最速下降)
  - [5.2 矩阵推广：Schatten-p 范数](#52-矩阵推广schatten-p-范数)
  - [5.3 求解约束优化问题](#53-求解约束优化问题)
  - [5.4 更新方向的性质分析](#54-更新方向的性质分析)
  - [5.5 与 Muon 和 SGD 的统一](#55-与-muon-和-sgd-的统一)
  - [5.6 高效近似计算](#56-高效近似计算)
- [6. 方法二：矩阵自然梯度（逆谱变换）](#6-方法二矩阵自然梯度逆谱变换)
  - [6.1 动机：让小奇异值主导更新](#61-动机让小奇异值主导更新)
  - [6.2 从 Adam 的逐元素逆到矩阵逆](#62-从-adam-的逐元素逆到矩阵逆)
  - [6.3 推导矩阵自然梯度方向](#63-推导矩阵自然梯度方向)
  - [6.4 数值稳定化](#64-数值稳定化)
  - [6.5 与 Fisher 预条件的联系](#65-与-fisher-预条件的联系)
- [7. 统一视角：谱变换函数族](#7-统一视角谱变换函数族)
- [8. 伪代码与实现方案](#8-伪代码与实现方案)
- [参考文献](#参考文献)

-----

## 1. 预备知识

### 1.1 矩阵范数与奇异值分解

设权重矩阵 $W \in \mathbb{R}^{n \times m}$，其奇异值分解（SVD）为

$$
W = U \Sigma V^\top = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

其中 $U \in \mathbb{R}^{n \times n}$，$V \in \mathbb{R}^{m \times m}$ 是正交矩阵，$\Sigma \in \mathbb{R}^{n \times m}$ 是对角矩阵，$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ 是正奇异值，$r = \text{rank}(W)$。

**Frobenius 范数**（Schatten-2 范数）：

$$
|W|*F = \sqrt{\sum*{i,j} W_{ij}^2} = \sqrt{\sum_{i=1}^{r} \sigma_i^2} = |\boldsymbol{\sigma}|_2
$$

**谱范数**（Schatten-$\infty$ 范数）：

$$
|W|*2 = \max*{|\mathbf{x}|=1} |W\mathbf{x}| = \sigma_1 = |\boldsymbol{\sigma}|_\infty
$$

**Schatten-p 范数**（一般定义）：对 $p \geq 1$，

$$
|W|*{S_p} = \left(\sum*{i=1}^{r} \sigma_i^p\right)^{1/p} = |\boldsymbol{\sigma}|_p
$$

特别地，$p=1$ 为核范数（trace norm），$p=2$ 为 Frobenius 范数，$p \to \infty$ 为谱范数 [1, 2]。

### 1.2 范数视角下的最速下降

对于向量参数 $\mathbf{w} \in \mathbb{R}^n$，最速下降的一般形式可由以下近端步骤定义 [3]：

$$
\mathbf{w}*{t+1} = \arg\min*{\mathbf{w}} \frac{|\mathbf{w} - \mathbf{w}_t|^2}{2\eta_t} + \mathcal{L}(\mathbf{w})
$$

其中 $|\cdot|$ 是某个向量范数。假设步长 $\eta_t$ 足够小使得一阶近似成立，令 $\Delta\mathbf{w} = \mathbf{w}_{t+1} - \mathbf{w}_t$，$\mathbf{g}*t = \nabla*{\mathbf{w}} \mathcal{L}(\mathbf{w}_t)$，则问题简化为：

$$
\Delta\mathbf{w}*{t+1} = \arg\min*{\Delta\mathbf{w}} \frac{|\Delta\mathbf{w}|^2}{2\eta_t} + \mathbf{g}_t^\top \Delta\mathbf{w} \tag{1}
$$

这等价于约束优化问题 [3, 4]：

$$
\min_{\Delta\mathbf{w}} ; \mathbf{g}_t^\top \Delta\mathbf{w} \quad \text{s.t.} \quad |\Delta\mathbf{w}| \leq \eta \tag{2}
$$

对于矩阵参数 $W \in \mathbb{R}^{n \times m}$，完全类比地有 [4]：

$$
\Delta W_{t+1} = \arg\min_{\Delta W} \frac{|\Delta W|^2}{2\eta_t} + \text{Tr}(G_t^\top \Delta W) \tag{3}
$$

即

$$
\min_{\Delta W} ; \text{Tr}(G_t^\top \Delta W) \quad \text{s.t.} \quad |\Delta W| \leq \eta \tag{4}
$$

其中 $G_t = \nabla_W \mathcal{L}(W_t)$ 是梯度矩阵，$|\cdot|$ 是某种矩阵范数。**选择不同的范数，就得到不同的优化器**——这是本文的核心框架。

-----

## 2. Muon 优化器的数学原理

### 2.1 从向量到矩阵：Sign 到 Msign

Muon（MomentUm Orthogonalized by Newton-schulz）的更新规则为 [5]：

$$
M_t = \beta M_{t-1} + G_t
$$
$$
W_t = W_{t-1} - \eta_t \left[\text{msign}(M_t) + \lambda W_{t-1}\right]
$$

其中 $\text{msign}$ 是**矩阵符号函数**（matrix sign function）。设 $M = U_M \Sigma_M V_M^\top$ 为动量的 SVD，则：

$$
\text{msign}(M) = U_M^{(r)} ; (V_M^{(r)})^\top \tag{5}
$$

其中 $U_M^{(r)}$ 和 $V_M^{(r)}$ 分别表示 $U_M$ 和 $V_M$ 的前 $r$ 列。

即**保留奇异向量，将所有奇异值替换为 1** [4]。

对比标量 sign 函数：$\text{sign}(x) = x / |x|$，将 $x$ 的绝对值替换为 1、仅保留符号。msign 是其精确的矩阵推广——在向量视角下 $\text{sign}(\mathbf{m}) = \mathbf{m}/|\mathbf{m}|$ 是 $\ell_2$ 归一化，在矩阵视角下 $\text{msign}(M) = U_M V_M^\top$ 是正交近似 [4]。

**恒等式**：msign 可以不通过 SVD 直接计算 [4]：

$$
\text{msign}(M) = M(M^\top M)^{-1/2} = (MM^\top)^{-1/2} M \tag{6}
$$

这与标量 $\text{sign}(x) = x(x^2)^{-1/2}$ 完全对应。

**最优正交近似**：当 $m = n = r$ 时 [4]：

$$
\text{msign}(M) = \arg\min_{O^\top O = I} |M - O|_F^2 \tag{7}
$$

即 msign(M) 是 $M$ 在正交矩阵集合中的 Frobenius 范数最近点。

### 2.2 约束优化视角

苏剑林在 [3] 中给出了 Muon 的约束优化推导。理想的优化器追求两个目标：**稳**（对模型扰动小）和**快**（loss 下降快）。形式化为：

$$
\min_{\Delta W} ; \text{Tr}(G^\top \Delta W) \quad \text{s.t.} \quad \rho(\Delta W) \leq \eta \tag{8}
$$

其中 $\rho(\Delta W)$ 是”稳”的度量。对于线性层 $\mathbf{y} = \mathbf{x} W$，输出扰动为 $|\Delta \mathbf{y}| = |\mathbf{x} \Delta W| \leq |\mathbf{x}| \cdot |\Delta W|_2$。因此**谱范数 $|\Delta W|_2$ 是控制输出扰动的最精准度量** [3]。

取 $\rho(\Delta W) = |\Delta W|_2$（谱范数），问题变为：

$$
\min_{\Delta W} ; \text{Tr}(G^\top \Delta W) \quad \text{s.t.} \quad |\Delta W|_2 \leq \eta \tag{9}
$$

**定理 1（Muon 是谱范数最速下降的解）**：问题 (9) 的解为：

$$
\Delta W = -\eta , \text{msign}(G) = -\eta , U^{(r)} (V^{(r)})^\top \tag{10}
$$

其中 $G = U \Sigma V^\top$ 是梯度的 SVD，$U^{(r)}$ 和 $V^{(r)}$ 分别为前 $r$ 列。

**证明**：将 $\Delta W$ 分解为”范数-方向”形式：$\gamma = |\Delta W|_2$，$\Phi = -\Delta W / |\Delta W|_2$（方向矩阵，满足 $|\Phi|_2 = 1$）。则：

$$
\min_{\Delta W} \text{Tr}(G^\top \Delta W) = \min_{\gamma \geq 0, |\Phi|_2 = 1} -\gamma , \text{Tr}(G^\top \Phi)
$$

$\gamma$ 的最优值为 $\gamma^* = \eta$（约束取等号），方向的最优值为：

$$
\Phi^* = \arg\max_{|\Phi|_2 = 1} \text{Tr}(G^\top \Phi) \tag{11}
$$

利用 $G = U \Sigma V^\top$ 展开：

$$
\text{Tr}(G^\top \Phi) = \text{Tr}(V \Sigma U^\top \Phi) = \text{Tr}(\Sigma U^\top \Phi V) = \sum_{i=1}^{r} \sigma_i (U^\top \Phi V)_{ii}
$$

其中 $\sigma_i > 0$ 是 $G$ 的奇异值。由于 $|\Phi|*2 = 1$，$U^\top \Phi V$ 的每个元素绝对值不超过 1。因此当且仅当 $(U^\top \Phi V)*{ii} = 1$ 对所有 $i$ 成立时，$\text{Tr}(G^\top \Phi)$ 取最大值。

这要求 $U^\top \Phi V = I$（在 $r \times r$ 子块上），即 $\Phi = U^{(r)} (V^{(r)})^\top = \text{msign}(G)$。

因此 $\Delta W^* = -\eta , \text{msign}(G)$。$\square$

当用动量 $M$ 替代 $G$ 时，就得到 Muon 的更新规则。

### 2.3 特征空间的对偶解释

苏剑林在 [6] 中进一步证明了一个深刻的对偶性质。对于线性层 $Y = XW$，参数更新 $W \leftarrow W - \eta , \text{msign}(\partial \mathcal{L}/\partial W)$，其中 $\partial \mathcal{L}/\partial W = X^\top (\partial \mathcal{L}/\partial Y)$。

输出特征的变化量为：

$$
\Delta Y = X(W - \eta , \text{msign}(X^\top \partial \mathcal{L}/\partial Y)) - XW = -\eta X , \text{msign}(X^\top \partial \mathcal{L}/\partial Y)
$$

利用公式 (6)：

$$
\Delta Y = -\eta X \cdot X^\top \frac{\partial \mathcal{L}}{\partial Y} \left(\frac{\partial \mathcal{L}}{\partial Y}^\top X X^\top \frac{\partial \mathcal{L}}{\partial Y}\right)^{-1/2}
$$

**关键观察**：若输入特征满足各向同性条件 $XX^\top \approx I$（这在使用 LayerNorm / RMSNorm 的 Transformer 中近似成立 [6]），则：

$$
\Delta Y \approx -\eta , \frac{\partial \mathcal{L}}{\partial Y} \left(\frac{\partial \mathcal{L}}{\partial Y}^\top \frac{\partial \mathcal{L}}{\partial Y}\right)^{-1/2} = -\eta , \text{msign}\left(\frac{\partial \mathcal{L}}{\partial Y}\right) \tag{12}
$$

即：**在各向同性条件下，Muon 在参数空间的谱范数最速下降，同时也是特征空间的谱范数最速下降** [6]。这是 Muon 在 pretraining 中高效的深层原因。

-----

## 3. RLVR 的优化动力学

### 3.1 参数更新稀疏性

Mukherjee et al. [7] 首先发现了 RL 微调中一个引人注目的现象：**参数更新稀疏性**（parameter update sparsity）。具体地：

- 在 7 种 RL 算法（PPO, GRPO, DPO, ORPO, KTO, SimPO, PRIME）和 10 个 LLM 上，RL 仅更新 **5%–30%** 的参数
- 子网络高度一致：跨随机种子、数据顺序、算法变体均可复现
- 仅微调该子网络即可复现全参微调效果

后续工作 [8] 进一步证实：SGD 在 RLVR 中可匹配 AdamW，且更新覆盖仅 0.02%–0.46% 的参数（比 AdamW 稀疏超过 1000 倍）。

### 3.2 Three-Gate Theory

Zhu et al. [9] 提出了 **Three-Gate Theory** 来机制性地解释 RLVR 的优化动力学。

**Gate I（KL Anchor）**：在线策略梯度方法天然施加隐式 KL 约束。设 $\pi_\theta$ 是当前策略，$\bar{q}_\beta(\cdot|x) \propto q(\cdot|x) \exp(R/\beta)$ 是改进后的目标分布，$\theta^+$ 是一步更新后的参数。则 [9, Proposition 3.1]：

$$
D_{\text{KL}}(\pi_{\theta^+} | \pi_\theta) \leq (1 + o(1)) , D_{\text{KL}}(\bar{q}*\beta | \pi*\theta)
$$

即每步策略漂移被 KL 约束限制。进一步 [9, Proposition 3.2]，若 Fisher 信息矩阵 $F(\theta) \geq \mu I$，则权重更新的范数有界：

$$
|\Delta W|_F \leq \sqrt{2K(1+o(1))}, \quad |\Delta W|_2 \leq \sqrt{\frac{2K}{\mu}(1+o(1))} \tag{13}
$$

其中 $K = D_{\text{KL}}(\pi_{\theta^+} | \pi_\theta)$。

**Gate II（Model Geometry）**：预训练模型的几何结构决定了 KL 约束下更新的**去向**。由 Wedin 的 $\sin\Theta$ 定理 [9, Theorem 3.3]：

$$
\max\left(|\sin\Theta(U_k(W_0), U_k(W_+))|*2, ; |\sin\Theta(V_k(W_0), V_k(W*+))|_2\right) \leq \frac{|\Delta W|_2}{\gamma_k} \tag{14}
$$

其中 $\gamma_k = \sigma_k(W_0) - \sigma_{k+1}(W_0)$ 是第 $k$ 奇异值 gap。对于 top-$k$ 主奇异子空间（$\gamma_k$ 大），子空间旋转被强约束；对于 tail 方向（$\gamma_k$ 小），约束弱，允许更多变化。因此**更新被引导至低曲率、谱保守的 off-principal 子空间** [9]。

**Gate III（Precision）**：bfloat16 有限精度过滤。由于 bfloat16 仅有 7 位尾数，权重 $W_{ij}$ 的可表示最小变化量（ULP）为 $\frac{1}{2} \text{ULP}*{\text{bf16}}(W*{ij})$ [9, Corollary 3.6]。在非偏好区域（principal 方向），微更新低于此阈值，在存储中被截断为零，造成**表观稀疏性** [9]。

### 3.3 Off-Principal 方向学习

三个 Gate 的联合效应导致一个核心结论 [9]：

> **RLVR 沿 off-principal 方向学习**：更新避开 principal weights（高幅值、高曲率），集中在 low-magnitude、low-curvature 的子空间。

实验验证 [9, Figure 5]：

- RL 更新 mask 与 principal weight mask 的重叠度**低于随机基线**（sub-random overlap）
- RL 更新 mask 与 low-magnitude mask 的重叠度**高于随机基线**（super-random overlap）
- RLVR 保持谱稳定：top-$k$ 奇异值几乎不变，主子空间旋转角 $< 2°$（vs SFT 的 $> 50°$）[9, Figure 4]

与此对比，SFT 走 principal 方向、扭曲谱结构 [9]。这意味着 **RL 和 SFT 处于本质不同的优化 regime**，需要不同的优化器设计。

-----

## 4. Muon 在 RL 中的结构性缺陷

### 4.1 各向同性更新与 Off-Principal 偏好的冲突

Muon 的更新方向 $\text{msign}(M) = U_M V_M^\top$ 将动量的**所有奇异值压成 1**。设动量 $M$ 的 SVD 为 $M = U_M \Sigma_M V_M^\top$，奇异值为 $\sigma_1^M \geq \sigma_2^M \geq \cdots \geq \sigma_r^M$，则 msign 实施的谱变换为：

$$
f_{\text{Muon}}(\sigma) = 1 \quad \forall \sigma > 0 \tag{15}
$$

这意味着：动量在不同奇异方向上的相对幅度信息**被完全抹去**。无论某个方向的梯度信号是强是弱，更新都等权重推进。

但 Three-Gate Theory（Section 3.2–3.3）告诉我们，RL 的有效梯度信号天然偏向 off-principal 方向。好的 RL 优化器应该**顺应并放大**这种偏好——**在 off-principal 方向给予更多权重，在 principal 方向抑制更新**。Muon 的各向同性 $f(\sigma) = 1$ 恰恰与此需求相矛盾。

### 4.2 谱结构保护的失败

RLVR 的一个关键特征是**保护预训练权重的谱结构**（Section 3.3）。设 $W_0$ 的 SVD 为 $W_0 = U_0 \Sigma_0 V_0^\top$。由 Wedin 定理 (14)，保护谱结构要求 $|\Delta W|_2$ 尽可能小。

Muon 的更新 $\Delta W = -\eta , \text{msign}(M)$ 满足 $|\Delta W|_2 = \eta$——这确实控制了谱范数。然而，更细致的分析表明，msign 产生的是一个**满秩正交矩阵**（当 $r = \min(n,m)$ 时），其对 $W_0$ 谱结构的扰动是**各向同性的**——所有奇异方向受到相同幅度的影响。

对于 RL，更理想的情况是：**在 $W_0$ 的 principal 方向（大 $\gamma_k$）几乎不动，在 off-principal 方向（小 $\gamma_k$）集中更新力量**。Muon 无法实现这种选择性保护。

### 4.3 Hopper 的经验修正及其局限

Hopper [10] 是一个最近提出的经验方法，其核心修改极其简单：

$$
\text{Hopper} = \text{Muon with } T_{\text{NS}} = 1 \text{ (instead of 5)} + \text{variance normalization}
$$

即将 Newton-Schulz 迭代从 5 步减少到 1 步。这意味着对动量矩阵仅做一次多项式变换：

$$
X_0 = M / |M|_F, \quad X_1 = aX_0 + bX_0(X_0^\top X_0) + cX_0(X_0^\top X_0)^2 \tag{16}
$$

其中 $a, b, c = (3.4445, -4.7750, 2.0315)$ [4]。

设 $X_0$ 的奇异值为 $\tilde{\sigma}_i = \sigma_i^M / |M|_F \in [0, 1]$，则一步迭代后的奇异值为：

$$
g(\tilde{\sigma}) = a\tilde{\sigma} + b\tilde{\sigma}^3 + c\tilde{\sigma}^5 \tag{17}
$$

这是一个 5 次多项式，将奇异值**部分地**推向 1（但不完全到达）。

**Hopper 的经验观察** [10]：在 0.5B 模型上用 GRPO 训练时，Hopper 在 200 步就匹配了 Adam 在 400 步的推理能力，且学到了不同的推理结构（如并行推理）。

**Hopper 的局限**：

1. **缺乏理论指导**：$g(\tilde{\sigma})$ 的形状完全由 Newton-Schulz 的系数 $(a, b, c)$ 决定，这些系数是为收敛到 msign 而优化的——不是为 RL 设计的
1. **不可控**：无法调节”正交化程度”——只有 $T=1$ vs $T=5$ 的离散选择
1. **不稳定**：作者发现 Hopper 需要 early stopping，长期训练会退化 [10]，并提出了”Hopper 探索 + Adam 收敛”的两阶段方案
1. **无闭式解释**：$g(\tilde{\sigma})$ 不对应任何已知的范数或距离度量下的最速下降方向

-----

## 5. 方法一：Schatten-p 范数下的最速下降

### 5.1 向量情形：$\ell_p$ 范数最速下降

作为铺垫，先回顾向量情形 [4]。问题 (2) 在 $\ell_p$ 范数下为：

$$
\boldsymbol{\varphi}^* = \arg\max_{|\boldsymbol{\varphi}|_p = 1} \mathbf{g}^\top \boldsymbol{\varphi} \tag{18}
$$

由 Holder 不等式 $\mathbf{g}^\top \boldsymbol{\varphi} \leq |\mathbf{g}|_q |\boldsymbol{\varphi}|_p$（其中 $\frac{1}{p} + \frac{1}{q} = 1$），等号成立的条件为 [4]：

$$
\varphi_i^* = \frac{1}{|\mathbf{g}|_q^{q/p}} , \text{sign}(g_i) |g_i|^{q/p} \tag{19}
$$

**验证**：直接使用 Lagrange 乘子法。定义 $L = \mathbf{g}^\top \boldsymbol{\varphi} - \frac{\lambda}{p}|\boldsymbol{\varphi}|_p^p$，对 $\varphi_i$ 求导：

$$
\frac{\partial L}{\partial \varphi_i} = g_i - \lambda |\varphi_i|^{p-2} \varphi_i = 0
$$

$$
\Rightarrow \varphi_i = \text{sign}(g_i) \left(\frac{|g_i|}{\lambda}\right)^{1/(p-1)} = \text{sign}(g_i) \left(\frac{|g_i|}{\lambda}\right)^{q-1}
$$

其中用了 $\frac{1}{p-1} = \frac{1}{p/(p-1) \cdot (p-1)/(p)} = q - 1$（由 $q = p/(p-1)$）。代入约束 $|\boldsymbol{\varphi}|_p = 1$ 确定 $\lambda$ 后，归一化得到 (19)。

**特殊情形**：

- $p = 2$（$q = 2$）：$\varphi_i^* \propto g_i$，即 SGD（梯度方向）
- $p \to \infty$（$q \to 1$）：$q/p \to 0$，$\varphi_i^* \to \text{sign}(g_i)$，即 SignSGD

### 5.2 矩阵推广：Schatten-p 范数

现在将问题推广到矩阵。在 Schatten-$p$ 范数约束下，问题 (4) 变为：

$$
\min_{\Delta W} ; \text{Tr}(G^\top \Delta W) \quad \text{s.t.} \quad |\Delta W|_{S_p} \leq \eta \tag{20}
$$

等价地，寻找方向矩阵：

$$
\Phi^* = \arg\max_{|\Phi|_{S_p} = 1} \text{Tr}(G^\top \Phi) \tag{21}
$$

### 5.3 求解约束优化问题

**定理 2（Schatten-p 范数最速下降方向）**：设 $G = U \Sigma V^\top$ 是梯度的 SVD，奇异值为 $\sigma_1 \geq \cdots \geq \sigma_r > 0$。令 $q = p/(p-1)$ 为共轭指数，$\alpha = q/p - 1 = (p-2)/(p-1)$。则问题 (21) 的解为：

$$
\Phi^* = \frac{1}{|\boldsymbol{\sigma}|_q^{q/p}} , U , \text{diag}(\sigma_1^{\alpha}, \sigma_2^{\alpha}, \ldots, \sigma_r^{\alpha}) , V^\top \tag{22}
$$

归一化后的更新方向（忽略常数因子）为：

$$
\boxed{\Delta W = -\eta \cdot U , \text{diag}(\sigma_i^{\alpha}) , V^\top / \text{RMS}, \quad \alpha = \frac{p-2}{p-1}} \tag{23}
$$

**证明**：

**第一步：化为对角问题。** 设 $\Phi$ 的 SVD 为 $\Phi = U_\Phi \Sigma_\Phi V_\Phi^\top$。利用 von Neumann 迹不等式 [11]：

$$
\text{Tr}(A^\top B) \leq \sum_{i=1}^{r} \sigma_i(A) \sigma_i(B)
$$

等号成立当且仅当 $A$ 和 $B$ 的左右奇异向量可同时对齐。因此：

$$
\text{Tr}(G^\top \Phi) = \text{Tr}(V \Sigma U^\top U_\Phi \Sigma_\Phi V_\Phi^\top) \leq \sum_{i=1}^{r} \sigma_i \cdot \sigma_i(\Phi)
$$

等号在 $U_\Phi = U$，$V_\Phi = V$ 时成立（即 $\Phi$ 与 $G$ 共享奇异向量）。因此最优 $\Phi$ 必然形如 $\Phi = U , \text{diag}(d_1, \ldots, d_r) , V^\top$，$d_i \geq 0$。

**第二步：求解奇异值。** 问题化为有限维优化：

$$
\max_{d_i \geq 0} \sum_{i=1}^{r} \sigma_i , d_i \quad \text{s.t.} \quad \left(\sum_{i=1}^{r} d_i^p\right)^{1/p} = 1 \tag{24}
$$

这正是向量 $\ell_p$ 范数约束下的最大化问题，与 Section 5.1 完全相同，只是将 $g_i$ 替换为 $\sigma_i$（非负）。由 (19)，最优解为：

$$
d_i^* = \frac{\sigma_i^{q/p}}{|\boldsymbol{\sigma}|_q^{q/p}} = \frac{\sigma_i^{q/p}}{\left(\sum_j \sigma_j^q\right)^{1/p}} \tag{25}
$$

**第三步：化简指数。** 由 $q = p/(p-1)$，有 $q/p = 1/(p-1)$。定义：

$$
\alpha \equiv q/p - 1 = \frac{1}{p-1} - 1 = \frac{1-(p-1)}{p-1} = \frac{2-p}{p-1}
$$

等等——这里需要小心。让我重新推导。

由 (19)，向量情形的最优解为 $\varphi_i^* \propto \text{sign}(g_i) |g_i|^{q/p}$。在矩阵情形中，$\sigma_i > 0$（奇异值非负），所以 $d_i^* \propto \sigma_i^{q/p}$。

但 $\Phi$ 的奇异值是 $d_i$，而 $|\Phi|_{S_p} = 1$ 的约束是 $\sum d_i^p = 1$。Lagrange 乘子法给出：

$$
\frac{\partial}{\partial d_i}\left[\sum_j \sigma_j d_j - \frac{\lambda}{p}\sum_j d_j^p\right] = \sigma_i - \lambda d_i^{p-1} = 0
$$

$$
\Rightarrow d_i = \left(\frac{\sigma_i}{\lambda}\right)^{1/(p-1)} = \left(\frac{\sigma_i}{\lambda}\right)^{q-1}
$$

注意 $1/(p-1) = q - 1$（由 $q = p/(p-1)$，故 $q - 1 = 1/(p-1)$）。

代入约束 $\sum d_i^p = 1$：

$$
\sum_i \left(\frac{\sigma_i}{\lambda}\right)^{p/(p-1)} = \sum_i \left(\frac{\sigma_i}{\lambda}\right)^q = 1
$$

$$
\Rightarrow \lambda^q = \sum_i \sigma_i^q \Rightarrow \lambda = |\boldsymbol{\sigma}|_q^{q/(q)} = |\boldsymbol{\sigma}|_q
$$

因此：

$$
d_i^* = \left(\frac{\sigma_i}{|\boldsymbol{\sigma}|_q}\right)^{q-1} = \left(\frac{\sigma_i}{|\boldsymbol{\sigma}|_q}\right)^{1/(p-1)} \tag{26}
$$

这里关键的指数是 $q - 1 = 1/(p-1)$。为了与更新方向的形式对应，我们将归一化常数吸收到学习率中，**更新方向的核心谱变换为**：

$$
f_{S_p}(\sigma) = \sigma^{1/(p-1)} \tag{27}
$$

**验证特殊情形**：

- $p = 2$（Frobenius 范数）：$f(\sigma) = \sigma^{1/(2-1)} = \sigma^1 = \sigma$。更新 $\propto U \Sigma V^\top = G$，即**普通梯度下降（SGD）**。 $\checkmark$
- $p \to \infty$（谱范数）：$f(\sigma) = \sigma^{1/(\infty - 1)} = \sigma^0 = 1$。更新 $\propto U V^\top = \text{msign}(G)$，即**Muon**。 $\checkmark$
- $p = 4$：$f(\sigma) = \sigma^{1/3}$。大奇异值被压缩（如 $\sigma=8 \to 2$），小奇异值相对放大（如 $\sigma=1 \to 1$），但大仍 $\geq$ 小。

因此最终的更新公式为：

$$
\boxed{\Delta W = -\eta \cdot \frac{U , \text{diag}!\left(\sigma_i^{1/(p-1)}\right) V^\top}{\text{RMS}!\left(U , \text{diag}!\left(\sigma_i^{1/(p-1)}\right) V^\top\right)}} \tag{28}
$$

$\square$

### 5.4 更新方向的性质分析

令 $\alpha = 1/(p-1)$。谱变换 $f(\sigma) = \sigma^\alpha$ 有以下关键性质：

**性质 1（单调性）**：当 $\alpha > 0$（即 $p > 1$），$f$ 是单调递增函数。因此大奇异值方向仍然获得更大的更新权重，但差距被压缩。

**性质 2（压缩比）**：两个奇异值 $\sigma_1 > \sigma_2$ 在变换后的比值为：

$$
\frac{f(\sigma_1)}{f(\sigma_2)} = \left(\frac{\sigma_1}{\sigma_2}\right)^\alpha
$$

- 当 $\alpha = 1$（SGD）：比值不变
- 当 $\alpha = 0$（Muon）：比值为 1（完全压平）
- 当 $0 < \alpha < 1$：$\left(\frac{\sigma_1}{\sigma_2}\right)^\alpha < \frac{\sigma_1}{\sigma_2}$，即差距被**幂律压缩**

**性质 3（对 RL 的匹配度）**：RL 梯度中，off-principal 方向对应的奇异值通常较小。$f(\sigma) = \sigma^\alpha$（$\alpha < 1$）相比 SGD 给予这些方向更大的相对权重，相比 Muon 保留了一定的方向区分度——这是一个 principled 的”部分正交化”。

### 5.5 与 Muon 和 SGD 的统一

Schatten-$p$ 最速下降提供了一个**连续参数族**，统一了 SGD 和 Muon：

|$p$     |$\alpha = 1/(p-1)$|$f(\sigma)$    |优化器 |
|--------|------------------|---------------|----|
|$2$     |$1$               |$\sigma$       |SGD |
|$3$     |$1/2$             |$\sqrt{\sigma}$|新方法 |
|$4$     |$1/3$             |$\sigma^{1/3}$ |新方法 |
|$\infty$|$0$               |$1$            |Muon|

RL 场景可能需要 $p \in (2, \infty)$ 的某个中间值（如 $p = 3$ 或 $p = 4$），具体最优值需要实验确定。

### 5.6 高效近似计算

直接 SVD 的复杂度为 $O(\min(n,m) \cdot nm)$，在大模型上不可接受。苏剑林在 [12] 中给出了关键的矩阵代数恒等式：

$$
U \Sigma^{2k} V^\top = \text{msign}(M) \cdot (M^\top M)^k \tag{29}
$$

$$
U \Sigma^{2k+1} V^\top = M \cdot (M^\top M)^k \tag{30}
$$

这意味着 $U f(\Sigma) V^\top$ 对**任意多项式** $f$ 都可以用 $M$、$\text{msign}(M)$、以及 $M^\top M$ 的幂来组合得到，无需显式 SVD [12]。

**实现方案**：对 $f(\sigma) = \sigma^\alpha$ 做多项式拟合：

$$
\sigma^\alpha \approx c_0 + c_1 \sigma + c_2 \sigma^2 + c_3 \sigma^3 + \cdots \tag{31}
$$

多项式系数 $c_k$ 可以通过 Chebyshev 逼近或最小二乘拟合预计算（依赖于 $\alpha$ 和奇异值的分布范围）。然后利用 (29)–(30)：

$$
U \left(\sum_k c_k \Sigma^k\right) V^\top = \sum_{k \text{ even}} c_k , \text{msign}(M) (M^\top M)^{k/2} + \sum_{k \text{ odd}} c_k , M (M^\top M)^{(k-1)/2} \tag{32}
$$

计算量为若干次矩阵乘法加上一次 msign（可复用 Muon 的 Newton-Schulz 基础设施）。

-----

## 6. 方法二：矩阵自然梯度（逆谱变换）

### 6.1 动机：让小奇异值主导更新

Section 5 的 Schatten-$p$ 方法使用 $f(\sigma) = \sigma^\alpha$（$\alpha > 0$），这是一个**单调递增**函数——大奇异值方向仍然获得更大权重，只是差距被缩小。

一个更激进的问题是：**能否让小奇异值方向获得比大奇异值方向更大的更新权重？** 即寻找一个**单调递减**的谱变换 $f$，使得：

$$
\sigma_1 > \sigma_2 \implies f(\sigma_1) < f(\sigma_2) \tag{33}
$$

从 RL 的角度看，如果动量中大奇异值方向与预训练权重的 principal 方向相关（Gate II 的推论），则反转奇异值大小关系恰好实现了”抑制 principal、放大 off-principal”。

### 6.2 从 Adam 的逐元素逆到矩阵逆

Adam [13] 的核心机制是逐元素自适应：

$$
\Delta w_i \propto \frac{m_i}{\sqrt{v_i} + \epsilon} \approx \frac{m_i}{|m_i|} = \text{sign}(m_i)
$$

其中 $m_i$ 是一阶矩，$v_i \approx m_i^2$ 是二阶矩。本质上，Adam 对每个参数做 $g_i \to g_i / |g_i|$——**梯度大的方向被除以大数变小，梯度小的方向被除以小数变大**。

这个操作可以理解为逐元素的”逆变换”：$f_{\text{Adam}}(|g_i|) = |g_i|^{-1} \cdot |g_i| = 1$，也就是 SignSGD。

现在我们要设计矩阵版本的逆变换。对于矩阵 $M = U_M \Sigma_M V_M^\top$，自然的推广是：

$$
f_{\text{inv}}(\sigma) = \sigma^{-\gamma}, \quad \gamma > 0 \tag{34}
$$

### 6.3 推导矩阵自然梯度方向

**定义**（矩阵逆谱变换方向）：

$$
\Phi_{\text{inv}} = U_M , \text{diag}(\sigma_i^{-\gamma}) , V_M^\top \tag{35}
$$

**当 $\gamma = 1$ 时的特殊结构**：

$$
\Phi_{\text{inv}} = U_M \Sigma_M^{-1} V_M^\top = M^{+} \tag{36}
$$

其中 $M^+$ 是 $M$ 的 Moore-Penrose 伪逆。利用公式 (6)：

$$
M^+ = V_M \Sigma_M^{-1} U_M^\top
$$

注意 (36) 的形式是 $U_M \Sigma_M^{-1} V_M^\top$，与 $M^+$ 的标准形式 $V_M \Sigma_M^{-1} U_M^\top$ 不同。让我们更仔细地分析 $\gamma = 1$ 的情形。

$$
U_M \Sigma_M^{-1} V_M^\top = \text{msign}(M) \cdot V_M \Sigma_M^{-1} V_M^\top = \text{msign}(M) \cdot (M^\top M)^{-1}  \cdot M^\top \cdot \text{msign}(M)
$$

实际上，更直接的等式是：

$$
U_M \Sigma_M^{-1} V_M^\top = (MM^\top)^{-1/2} \cdot M \cdot (M^\top M)^{-1/2} \tag{37}
$$

**验证**：$(MM^\top)^{-1/2} M (M^\top M)^{-1/2} = U \Sigma^{-1/2} U^\top \cdot U \Sigma V^\top \cdot V \Sigma^{-1/2} V^\top = U \Sigma^{-1/2} \Sigma \Sigma^{-1/2} V^\top = U \Sigma^0 V^\top = U V^\top$。

等等，这给出 msign(M)，不是我们想要的。让我重新推导。

对于一般的幂函数 $f(\sigma) = \sigma^\alpha$，我们有：

$$
U \Sigma^\alpha V^\top = U \Sigma^{\alpha} V^\top
$$

当 $\alpha = -1$ 时：

$$
U \Sigma^{-1} V^\top
$$

利用恒等式 (29) 的推广。注意 $\Sigma^{-1}$ 不能直接用 $M^\top M$ 的正幂表示，需要矩阵求逆。但我们可以写成：

$$
U \Sigma^{-1} V^\top = \text{msign}(M) \cdot (M^\top M)^{-1} \cdot \Sigma \cdot \text{msign}(M)^\top \cdot \text{msign}(M)
$$

这太复杂了。更简洁的表达是：

利用 $M^\top M = V \Sigma^2 V^\top$，我们有 $(M^\top M)^{-1} = V \Sigma^{-2} V^\top$。因此：

$$
M (M^\top M)^{-1} = U \Sigma V^\top V \Sigma^{-2} V^\top = U \Sigma^{-1} V^\top \tag{38}
$$

这就是我们要的！即：

$$
\boxed{U \Sigma^{-1} V^\top = M (M^\top M)^{-1}} \tag{39}
$$

**验证**：$M(M^\top M)^{-1} = U\Sigma V^\top (V\Sigma^2 V^\top)^{-1} = U\Sigma V^\top V\Sigma^{-2}V^\top = U\Sigma^{-1}V^\top$。 $\checkmark$

更一般地，对于任意 $\alpha$：

$$
U \Sigma^{2k+1} V^\top = M(M^\top M)^k \tag{40}
$$

当 $k = -1$：$U \Sigma^{-1} V^\top = M (M^\top M)^{-1}$。 $\checkmark$

### 6.4 数值稳定化

$f(\sigma) = \sigma^{-\gamma}$ 在 $\sigma \to 0$ 时发散，必须进行稳定化处理。

**方案 A（$\epsilon$-截断）**：

$$
f_\epsilon(\sigma) = (\sigma + \epsilon)^{-\gamma}, \quad \epsilon > 0 \tag{41}
$$

等价于将 $M$ 替换为 $M + \epsilon I$（在适当意义下）。

**方案 B（软化伪逆）**：利用 Tikhonov 正则化思想：

$$
U \Sigma^{-1} V^\top \approx M(M^\top M + \epsilon^2 I)^{-1} \tag{42}
$$

**验证**：$M(M^\top M + \epsilon^2 I)^{-1} = U\Sigma V^\top (V(\Sigma^2 + \epsilon^2 I)V^\top)^{-1} = U\Sigma(\Sigma^2+\epsilon^2 I)^{-1}V^\top = U , \text{diag}!\left(\frac{\sigma_i}{\sigma_i^2+\epsilon^2}\right) V^\top$。

当 $\epsilon \to 0$，$\frac{\sigma}{\sigma^2+\epsilon^2} \to 1/\sigma$；当 $\sigma \to 0$，$\frac{\sigma}{\sigma^2+\epsilon^2} \to 0$（而非发散）。这正是理想的稳定化行为。

因此，稳定化后的逆谱变换为：

$$
\boxed{\Phi_{\text{inv}} = M(M^\top M + \epsilon^2 I)^{-1} / \text{RMS}} \tag{43}
$$

其谱变换函数为：

$$
f_{\text{inv}}(\sigma) = \frac{\sigma}{\sigma^2 + \epsilon^2} \tag{44}
$$

$f_{\text{inv}}$ 的关键性质：

- 在 $\sigma = \epsilon$ 处取最大值 $\frac{1}{2\epsilon}$
- 当 $\sigma \gg \epsilon$ 时，$f \approx 1/\sigma$（递减）
- 当 $\sigma \ll \epsilon$ 时，$f \approx \sigma/\epsilon^2$（递增）

因此 $f_{\text{inv}}$ 是一个**先增后减**的钟形函数——**中等大小的奇异值获得最大权重，极大和极小的都被抑制**。如果 $\epsilon$ 设置在 principal/off-principal 的分界附近，这恰好能选择性地放大 off-principal 信号。

### 6.5 与 Fisher 预条件的联系

矩阵自然梯度 (43) 与经典的自然梯度 / Fisher 信息矩阵预条件有深层联系。

在参数空间中，自然梯度 [14] 为 $\Delta\theta = -F(\theta)^{-1} \nabla\mathcal{L}$，其中 $F(\theta) = \mathbb{E}[\nabla\log\pi_\theta \nabla\log\pi_\theta^\top]$ 是 Fisher 信息矩阵。

对于矩阵参数 $W$，如果我们将 $M^\top M$（动量的 Gram 矩阵）视为 Fisher 信息在该参数块上的代理（这在一阶矩和二阶矩近似下是合理的），则：

$$
\Phi_{\text{inv}} = M(M^\top M + \epsilon^2 I)^{-1} \approx M \cdot F_W^{-1}
$$

这与 K-FAC [15] 等二阶方法的 Kronecker 近似在精神上一致，但我们的方法直接操作矩阵的奇异值结构，保留了方向间的耦合——**不像 Adam 那样在逐元素层面打散矩阵结构**。

**对比 Adam**：

- Adam：$\Delta W_{ij} \propto m_{ij} / \sqrt{v_{ij}}$——逐元素操作，将 $W$ 视为一个长向量
- 矩阵自然梯度：$\Delta W \propto M(M^\top M + \epsilon^2 I)^{-1}$——在矩阵奇异值层面操作，保留行列间的耦合关系

-----

## 7. 统一视角：谱变换函数族

上述所有方法可以统一为**谱变换**框架。给定动量 $M = U_M \Sigma_M V_M^\top$，更新方向为：

$$
\Phi = U_M , \text{diag}(f(\sigma_1), \ldots, f(\sigma_r)) , V_M^\top / \text{RMS} \tag{45}
$$

不同的 $f$ 对应不同的优化器：

|方法          |$f(\sigma)$                   |参数                |大 $\sigma$ / 小 $\sigma$ 比值          |理论基础                  |
|------------|------------------------------|------------------|------------------------------------|----------------------|
|SGD         |$\sigma$                      |无                 |$\sigma_1/\sigma_r$ (不变)            |Schatten-2 最速下降       |
|Schatten-$p$|$\sigma^{1/(p-1)}$            |$p \in (2,\infty)$|$(\sigma_1/\sigma_r)^{1/(p-1)}$ (压缩)|Schatten-$p$ 最速下降     |
|Muon        |$1$                           |无                 |$1$ (抹平)                            |Schatten-$\infty$ 最速下降|
|矩阵自然梯度      |$\sigma/(\sigma^2+\epsilon^2)$|$\epsilon > 0$    |$\approx \sigma_r/\sigma_1$ (反转)    |Fisher 预条件推广          |

**对 RL 的适用性分析**：

- **SGD**（$f = \sigma$）：保留梯度相对幅度，不做任何矩阵感知的处理。Mukherjee et al. [8] 已证明在 RL 中表现不差，但缺乏结构性增强
- **Schatten-$p$**（$f = \sigma^\alpha$，$0 < \alpha < 1$）：幂律压缩，是 Muon 和 SGD 之间的可控插值。**适合”温和放大 off-principal”场景**
- **Muon**（$f = 1$）：完全抹平，与 RL 的 off-principal 偏好冲突
- **矩阵自然梯度**（$f = \sigma/(\sigma^2+\epsilon^2)$）：反转大小关系。**适合”激进放大 off-principal”场景**

-----

## 8. 伪代码与实现方案

### 8.1 Schatten-p 优化器

```
Algorithm: Schatten-p Steepest Descent Optimizer
-----------------------------------------------
Input: W ∈ R^{n×m}, learning rate η, momentum β, weight decay λ
       p ∈ (2, ∞)          -- interpolation parameter
State: M (momentum buffer, initialized to 0)

Precompute: α = 1/(p-1)

For each step t:
    G_t = ∇_W L(W_{t-1})                   -- compute gradient
    M_t = β M_{t-1} + G_t                  -- accumulate momentum

    -- Compute spectral transform U diag(σ_i^α) V^T
    -- Option A: via SVD (exact)
    U, Σ, V^T = SVD(M_t)
    Φ = U · diag(σ_i^α) · V^T

    -- Option B: via polynomial approximation (practical)
    -- Fit σ^α ≈ Σ_k c_k σ^k on [0, σ_max/||M||_F]
    -- Use identities (29)-(30) to assemble from msign(M) and (M^T M)^k

    Φ = Φ / RMS(Φ)                         -- RMS alignment
    W_t = W_{t-1} - η (Φ + λ W_{t-1})      -- update with weight decay
```

### 8.2 矩阵自然梯度优化器

```
Algorithm: Matrix Natural Gradient Optimizer
--------------------------------------------
Input: W ∈ R^{n×m}, learning rate η, momentum β, weight decay λ
       ε > 0               -- stabilization constant
State: M (momentum buffer, initialized to 0)

For each step t:
    G_t = ∇_W L(W_{t-1})                   -- compute gradient
    M_t = β M_{t-1} + G_t                  -- accumulate momentum

    -- Compute M (M^T M + ε² I)^{-1}
    -- Option A: via SVD (exact)
    U, Σ, V^T = SVD(M_t)
    Φ = U · diag(σ_i / (σ_i² + ε²)) · V^T

    -- Option B: via matrix solve (practical, avoids SVD)
    -- Solve the linear system: Φ · (M_t^T M_t + ε² I) = M_t
    -- i.e., Φ = M_t · (M_t^T M_t + ε² I)^{-1}
    -- Can use conjugate gradient or Neumann series approximation

    Φ = Φ / RMS(Φ)                         -- RMS alignment
    W_t = W_{t-1} - η (Φ + λ W_{t-1})      -- update with weight decay
```

### 8.3 对比一览

```
                  Muon              Schatten-p          Matrix Nat. Grad.
                  ──────────────     ──────────────      ──────────────
核心操作          msign(M)           U σ^α V^T           M(M^T M + ε²I)^{-1}
谱变换 f(σ)       1                  σ^{1/(p-1)}         σ/(σ²+ε²)
超参              无                 p ∈ (2,∞)           ε > 0
大σ vs 小σ        等权重              大>小(压缩)          小>大(反转)
理论基础          Schatten-∞ 最速下降  Schatten-p 最速下降   Fisher 预条件推广
计算复杂度        ~Muon (NS迭代)      ~Muon + 矩阵幂       ~Muon + 矩阵求逆
额外内存          无                 无                   无 (若用迭代法)
RL适用性          差(各向同性)        中(可控压缩)          强(off-principal放大)
```

-----

## 参考文献

[1] R. Bhatia. *Matrix Analysis*. Springer, 1997.

[2] R. A. Horn and C. R. Johnson. *Matrix Analysis*. Cambridge University Press, 2nd edition, 2013.

[3] 苏剑林. “Muon续集：为什么我们选择尝试Muon？” 科学空间, Feb. 2025. https://kexue.fm/archives/10739

[4] 苏剑林. “Muon优化器赏析：从向量到矩阵的本质跨越.” 科学空间, Dec. 2024. https://kexue.fm/archives/10592

[5] K. Jordan. “Muon: An optimizer for hidden layers in neural networks.” Blog post, 2024.

[6] 苏剑林. “为什么我们偏爱各向同性？基于最速下降的理解.” 科学空间, Jan. 2026. https://kexue.fm/archives/11549

[7] S. Mukherjee, L. Yuan, D. Hakkani-Tur, H. Peng. “Reinforcement Learning Finetunes Small Subnetworks in Large Language Models.” NeurIPS 2025. arXiv:2505.11711.

[8] S. Mukherjee, L. Yuan, P. Jayasinha, D. Hakkani-Tur, H. Peng. “Do We Need Adam? Surprisingly Strong and Sparse Reinforcement Learning with SGD in LLMs.” arXiv:2602.07729, 2026.

[9] H. Zhu, Z. Zhang, H. Huang, et al. “The Path Not Taken: RLVR Provably Learns Off the Principals.” arXiv:2511.08567, 2025.

[10] bird-of-paradise. “Hopper – partial orthogonalization changes early reasoning behavior in RL.” Hugging Face Forums, Feb. 2026. https://discuss.huggingface.co/t/hopper-partial-orthogonalization-changes-early-reasoning-behavior-in-rl/173133

[11] J. von Neumann. “Some matrix-inequalities and metrization of matric-space.” *Tomsk Univ. Rev.*, 1937.

[12] 苏剑林. “通过msign来计算奇异值裁剪mclip（上）.” 科学空间, Jun. 2025. https://kexue.fm/archives/11006

[13] D. P. Kingma and J. Ba. “Adam: A Method for Stochastic Optimization.” ICLR 2015. arXiv:1412.6980.

[14] S. Amari. “Natural Gradient Works Efficiently in Learning.” *Neural Computation*, 10(2), 1998.

[15] J. Martens and R. Grosse. “Optimizing Neural Networks with Kronecker-Factored Approximate Curvature.” ICML 2015.

[16] J. Liu, J. Su, et al. “Muon is Scalable for LLM Training.” arXiv:2502.16982, 2025.

[17] 苏剑林. “从谱范数梯度到新式权重衰减的思考.” 科学空间, Dec. 2024. https://kexue.fm/archives/10648

[18] T. Joo, W. Xia, C. Kim, M. Zhang, E. Ie. “On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.” arXiv:2602.15322, 2026.

[19] C. Si, D. Zhang, W. Shen. “AdaMuon: Adaptive Muon Optimizer.” arXiv:2507.11005, 2025.

[20] 苏剑林. “基于流式幂迭代的Muon实现：1. 初识.” 科学空间, Mar. 2026. https://kexue.fm/archives/11588
