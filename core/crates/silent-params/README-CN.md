# silent-params

`silent-params` 是 SILENT 的统一参数语义层。

英文版请见: [`README.md`](README.md)

它不是 BFV、HSS、RLWE 或未来算法的执行层，也不是简单的“配置文件集合”。它承担的是更基础的职责:

- 定义参数语言
- 定义参数对象
- 定义验证规则
- 定义稳定身份
- 定义可版本化的 presets

一句话说清楚:

- `silent-params` 负责描述“参数是什么意思”
- `silent-rlwe`、`silent-fhe`、`silent-hss` 等运行时 crate 负责描述“这些参数如何执行”

## 这个包为什么存在

如果一个密码学库没有独立的参数层，代码通常会慢慢变得很脆弱:

- 每个算法各自长出一套参数结构
- 同一个概念在不同 crate 里有不同名字
- 验证逻辑分散、重复，甚至缺失
- 安全性假设写在脑子里，不写在类型里
- benchmark 参数和生产参数不是同一种表达

SILENT 明确不想走到这一步。

所以 `silent-params` 的意义不是“多拆一个包”，而是把参数设计提升为架构中心。这样做的收益很直接:

- 正确性更强: 非法参数组合可以在运行前被拒绝
- 可审计性更强: 参数含义集中表达，不靠隐含约定
- 可复现性更强: 通过 `ParamsId` 可以稳定标识一个参数集合
- 扩展性更强: 新算法可以接入统一参数语言
- 可维护性更强: 运行时实现专注执行，不再承担参数政策职责

## 设计目标

这个 crate 的设计目标很明确。

## 1. canonical params 必须先于 runtime params

SILENT 里每一种 scheme 都应该先有一个稳定、可验证、不可变的参数对象，用来表达“这个参数集合在语义上是什么”。

当前第一批核心对象包括:

- `RingParams`
- `RlweParams`
- `BfvParams`
- `CkksParams`
- `HssParams`
- `PqcKemParams`
- `PqcSigParams`

这些对象才是源头真相。运行时结构应该从它们派生，而不是反过来让运行时对象主导参数语义。

## 2. 强类型优先，不接受满地裸整数

这个 crate 从一开始就尽量避免把密码学参数当作普通 `usize` 或 `u64` 在系统里裸传。

因此这里定义了很多 newtype:

- `LogN`
- `RingDim`
- `ModulusBits`
- `PlaintextModulus`
- `ScaleBits`
- `MultiplicativeDepth`
- `ShareModulusBits`

这样做的目的不是“好看”，而是减少语义混淆，避免把不同含义但底层类型相同的值混在一起。

## 3. validate 不是辅助函数，而是契约的一部分

所有 canonical 参数集合都实现 `ParameterSet`，其中 `validate()` 是核心接口之一。

这意味着:

- 参数验证不是调用方“有空就做”
- 参数验证不是分散在不同实现里的隐性逻辑
- 参数验证是参数模型本身的一部分

## 4. 参数需要稳定身份

每个 canonical parameter set 都可以生成一个 `ParamsId([u8; 32])`。

这个 ID 不是拿 `Debug` 字符串去 hash 出来的，而是基于稳定 canonical encoding 生成的。

它的价值在后面会越来越大:

- 参数缓存
- 工件追踪
- 兼容性判断
- benchmark 结果复现
- 未来序列化/注册表系统

## 5. presets 必须天然支持版本化

这个 crate 里已经有:

- `presets/toy.rs`
- `presets/dev.rs`
- `presets/current.rs`

这不是为了凑结构，而是为了让 SILENT 以后可以安全演进默认参数，同时依然能精确引用旧参数集。

换句话说，SILENT 不应该只有“当前默认值”，还应该有“这个默认值属于哪个时代、哪一版”的概念。

## 设计来源与 SILENT 的取舍

SILENT 不是简单照搬某一个现有库。

我们的思路是吸收几个成熟项目各自最强的部分，再组合成适合 SILENT 的结构。

### 借鉴来源

- OpenFHE: `SecurityLevel` 的表达方式，以及基于标准表的自动校验思路
- Lattigo: literal 参数描述到 validated immutable params 的路径
- TFHE-rs: 强类型 newtype，反对裸整数乱飞

### SILENT 自己的设计选择

- `ParamsId` 作为 canonical 身份
- `ParameterSet` 作为跨 scheme 的统一契约
- versioned presets 作为正式结构，而不是散落常量
- HSS 拥有独立参数模型，而不是被当成 RLWE/BFV 的薄包装

## 这里到底放什么

当前 crate 主要由这些模块组成:

- `security.rs`: 安全级别定义
- `distribution.rs`: 分布类型定义
- `scheme.rs`: scheme family 分类
- `newtypes.rs`: 强类型参数标量
- `ids.rs`: `ParamsId`、canonical encoding、`ParameterSet`
- `ring.rs`: ring 级别参数与约束
- `rlwe.rs`: RLWE 级别模链、prime 生成与验证
- `fhe.rs`: BFV 与 CKKS 参数对象
- `hss.rs`: HSS 特有参数对象与 bound-aware 验证
- `pqc.rs`: 未来 PQC 参数对象
- `standard.rs`: HE Standard 风格查表接口
- `presets/`: 参数预设集合

## 这一层负责什么，不负责什么

`silent-params` 负责的是:

- 表达参数语义
- 验证结构和安全约束
- 描述或生成模链需求
- 提供稳定参数身份
- 组织预设参数集

它不负责的是:

- 密文运算
- 密钥生成实现
- NTT 或 RNS 的执行
- 运行时缓存和表结构
- 网络/磁盘序列化格式

这一点非常重要。这个 crate 是参数契约层，不是执行引擎层。

## canonical params 和 runtime params 的关系

这是整个 SILENT 架构里最关键的分层之一。

canonical params 的特点:

- 对人可读
- 含义稳定
- 可以提前验证
- 适合做 presets、registry、兼容性边界

runtime params 的特点:

- 面向执行
- 可以带缓存、表、派生状态
- 不同实现可以有不同内部布局
- 只要语义一致，就可以对应同一个 canonical parameter set

例如 BFV 在 `silent-fhe` 里可以从一个 `BfvParams` 出发，派生出具体 runtime 结构。这种分层把两件事分开了:

- “这个参数集合想表达什么”
- “这个实现打算怎样执行它”

这正是地基稳定的关键。

## 为什么 HSS 不能只复用 BFV/RLWE 参数

HSS 不是普通 FHE 项目里的附属模块，所以它不能只靠 `ring_dim + q` 这种表达活着。

`HssParams` 当前明确包含:

- `share_modulus_bits`
- `max_linear_terms`
- `max_rmult_depth`
- `fixed_point_scale_bits`
- `reconstruction_bound_bits`
- `correctness_margin_bits`

原因很直接:

HSS 的正确性不只取决于 ring 和 modulus chain，还取决于 share、reconstruction、fixed-point scale、overflow margin 这些 bound-aware 约束。

如果以后 SILENT 要支持:

- 金融阈值密态计算
- 私密聚合
- 规则匹配
- 多方 bound-aware 协议

那么 HSS 参数必须是独立的一等模型，而不是“顺手塞在 BFV 参数旁边”。

## 这层对 CKKS 和 TFHE 的意义

是的，这一层对后续添加 `CKKS` 和 `TFHE` 非常关键。

但它的价值不只是“加新算法更方便”，而是“加新算法时不会把库的架构搞乱”。

### 对 CKKS 的意义

CKKS 和 BFV 一样共享 RLWE 基础，所以天然适合放进这套参数语言里。

它可以复用:

- `RingParams`
- `RlweParams`
- `SecurityLevel`
- `DistributionType`
- 标准表查验逻辑

再在自己这一层补充:

- `ScaleBits`
- depth/level 预算
- rescale 相关约束

当前 `CkksParams` 已经是这条路线上的第一步。

### 对 TFHE 的意义

TFHE 不应该被强行塞进 BFV/CKKS 的世界观里，但它完全应该加入 SILENT 的统一参数体系。

比较合理的未来方向是补出专门类型，例如:

- `LweParams`
- `GlweParams`
- `BootstrapParams`
- `KeyswitchParams`
- `TfheParams`

这样既能共享 SILENT 的参数哲学，又不会扭曲 TFHE 自己的数学结构。

## 当前几个核心概念

## `SecurityLevel`

安全级别现在不是只保留 classical 三档，而是明确支持:

- `Toy`
- `Classical128`
- `Classical192`
- `Classical256`
- `Quantum128`
- `Quantum192`
- `Quantum256`
- `NotSet`

这说明 SILENT 从一开始就不把“安全级别”理解成单一视角下的数字，而是一个有语义的分类。

## `ParameterSet`

所有 canonical 参数对象都统一实现:

```rust
pub trait ParameterSet {
    fn name(&self) -> &'static str;
    fn scheme_family(&self) -> SchemeFamily;
    fn security_level(&self) -> SecurityLevel;
    fn params_id(&self) -> ParamsId;
    fn validate(&self) -> Result<(), ParamError>;
}
```

这意味着无论是 BFV、CKKS、HSS，还是以后加入的 PQC/TFHE 参数对象，都可以被统一地:

- 注册
- 校验
- 标识
- 比较
- 编排 presets

## `ParamsId`

`ParamsId` 的意义不是“给参数搞个 hash 玩玩”，而是把参数身份稳定下来。

我们希望以后一个参数集合可以被明确地说成:

- 它叫什么
- 它属于什么 scheme family
- 它的语义内容是什么
- 它的稳定 ID 是什么

这样做对实验复现、缓存一致性、兼容性边界都非常重要。

## 标准表与安全预算

`standard.rs` 里现在有 HE Standard 风格的接口:

- `find_max_log_q(...)`
- `find_min_ring_dim(...)`

第一版只先落了 ternary secret + classical security 的查表逻辑，但接口没有被写死在这个版本上。

这代表 SILENT 的态度是:

- 第一阶段先保守、先准确
- 但 API 从一开始就为后续表扩展留好边界

## 推荐使用方式

这层建议的使用路径是:

1. 选择或构造一个 canonical parameter set
2. 调用 `validate()`
3. 获取 `ParamsId`
4. 派生到运行时结构
5. 在执行层里运行算法

例如:

```rust
use silent_params::{BfvParams, ParameterSet};

fn assert_valid(params: &BfvParams) {
    params.validate().expect("invalid BFV params");
    let _id = params.params_id();
}
```

运行时 crate 再根据这些 canonical params 去构建自己的执行对象。

## 从架构角度看，这一层真正的意义

如果只记住一句话，那就是:

`silent-params` 的意义，是把“参数设计”从实现细节提升为架构中心。

这层存在以后，SILENT 才更有可能成长为一个多算法、可扩展、可审计、可演进的密码库，而不是慢慢变成一堆彼此不兼容的构造器和隐藏假设。 
