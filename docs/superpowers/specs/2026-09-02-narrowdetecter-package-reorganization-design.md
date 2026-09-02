# NarrowDetecter 功能化重组设计

日期：2026-09-02

## 1. 背景

`NarrowDetecter` 当前已经包含冠脉中心线提取、分支命名、横截面区域生长、狭窄检测、straightened/stretched CPR、结果序列化和多种可视化功能。现有实现可以运行，但模块主要平铺在包根目录，且若干文件同时承担算法、文件读写、流程编排和可视化职责。

本次工作只重组现有功能并增加综合工作流入口，不改变狭窄判定方法、CPR 几何定义、现有默认参数或医学图像输出内容。重组后应更容易单独理解、调用和维护狭窄检测、CPR 以及二者联动流程。

## 2. 目标

1. 按功能拆分代码，使每个子包职责清晰。
2. 提供三个正式工作流函数：`run_detection`、`run_cpr` 和 `run_combined`。
3. 综合流程直接复用狭窄检测阶段得到的中心线、分支及平行运输框架，不重复提取中心线。
4. 保留现有狭窄检测、解剖命名、straightened/stretched CPR 和狭窄标记行为。
5. 输出目录完全由调用者指定；程序只创建明确指定的目录，不推导或增加子目录。
6. 删除没有后续算法用途的 CSV 输出。

## 3. 非目标

1. 不修改区域生长、参考平面选择、狭窄率或信号强度指标的计算公式。
2. 不修改中心线树提取、分支切分和解剖命名规则。
3. 不修改 straightened 或 stretched CPR 的采样几何、背景填充方式和标记样式。
4. 不增加命令行参数解析器或命令行入口。
5. 不保留旧的 `NarrowDetecter.cli` 和 `NarrowDetecter.cpr_cli` 导入兼容性。
6. 不在 `NarrowDetecter` 包内保留测试目录或测试文件。

## 4. 目录结构

目标目录结构如下：

```text
NarrowDetecter/
  __init__.py
  api/
    __init__.py
    detection.py
    cpr.py
    combined.py
  detection/
    __init__.py
    detector.py
    cross_section.py
  cpr/
    __init__.py
    generator.py
  centerline/
    __init__.py
    extraction.py
    anatomy.py
  visualization/
    __init__.py
    detection.py
    cpr.py
    volumes.py
  io/
    __init__.py
    images.py
    results.py
  models/
    __init__.py
    detection.py
    cpr.py
    combined.py
  docs/
    superpowers/
      specs/
  README.md
  requirements.txt
```

### 4.1 `api`

`api` 只负责工作流编排：校验函数参数、读取输入、调用核心算法、保存结果和返回数据对象。算法细节不得放入该子包。

- `api/detection.py`：实现 `run_detection`。
- `api/cpr.py`：实现 `run_cpr`。
- `api/combined.py`：实现 `run_combined`。

### 4.2 `detection`

`detection` 负责横截面采样后的管腔分割、质量判断、参考值计算、狭窄候选合并和信号强度统计。

- `detection/detector.py`：狭窄检测主流程、参考直径及候选生成。
- `detection/cross_section.py`：横截面采样和区域生长。

### 4.3 `cpr`

`cpr` 负责 CPR 数值生成，不承担文件读写或绘图职责。

- `cpr/generator.py`：中心线重采样、平行运输框架以及 straightened/stretched CPR 生成。
- `cpr/__init__.py`：按需重新导出稳定的底层 CPR 生成函数，避免调用方依赖内部文件名。

### 4.4 `centerline`

`centerline` 负责中心线树构建和解剖结构处理。

- `centerline/extraction.py`：骨架图、root 选择、路径提取、平滑、重采样和框架计算。
- `centerline/anatomy.py`：分支切分、主干选择及 LAD/LCX/RCA 分支命名。

### 4.5 `visualization`

可视化按输出类型拆分，避免单个文件同时处理检测曲线、CPR 和三维 NIfTI 标记。

- `visualization/detection.py`：直径/强度曲线和狭窄局部视图。
- `visualization/cpr.py`：CPR 图像、二维中心线和狭窄标记。
- `visualization/volumes.py`：中心线标签和狭窄标记 NIfTI。

### 4.6 `io`

- `io/images.py`：SimpleITK 图像、标签和主动脉 mask 读取与几何检查。
- `io/results.py`：检测 JSON 的保存和读取，以及 CPR 标记所需候选解析。

CSV 写入逻辑及 `csv` 依赖从该层删除。

### 4.7 `models`

模型层只包含数据类和轻量序列化，不依赖算法、SimpleITK 或 Matplotlib。

- `models/detection.py`：`DetectorConfig`、横截面测量、分支、候选、血管和 `DetectionResult`。
- `models/cpr.py`：`CPRConfig` 和 `CPRResult`。
- `models/combined.py`：`CombinedResult`。

## 5. 公开接口

包根目录提供以下正式导入方式：

```python
from NarrowDetecter import run_detection, run_cpr, run_combined
```

根目录还可以导出调用三个工作流所必需的配置类和结果类。中心线、横截面和 CPR 生成等底层函数仍可从对应功能子包导入，但不作为包根目录的正式工作流接口。

旧的 `cli.py` 和 `cpr_cli.py` 将被删除，不提供兼容转发文件。

### 5.1 `run_detection`

`run_detection` 保持当前函数参数及 `DetectionResult` 返回类型。它读取标签、强度图像和可选主动脉 mask，运行检测，保存 JSON，并按参数决定是否保存检测可视化。

输出目录不存在时创建。输出目录由 `output_dir` 明确指定，函数不增加子目录。

### 5.2 `run_cpr`

`run_cpr` 保持当前直接函数参数及 `list[CPRResult]` 返回类型。它继续支持两种互斥的中心线来源：

1. 调用者直接提供有序中心线和可选框架。
2. 调用者提供冠脉标签，由函数内部提取中心线树。

它继续支持 `straightened`、`stretched`、多角度、侧支开关、`stenosis_result_path` 和只返回内存结果的调用方式。`save_outputs=False` 时，`output_dir` 可以是 `None`。

### 5.3 `run_combined`

综合接口定义为：

```python
run_combined(
    label_path,
    image_path,
    detection_output_dir,
    cpr_output_dir,
    detection_config=None,
    cpr_config=None,
    aorta_mask_path=None,
    print_summary=True,
    save_detection_visualizations=True,
    marker_radius_mm=1.5,
    centerline_label_radius_mm=0.6,
) -> CombinedResult
```

图像路径、两个输出目录和主动脉 mask 路径为直接函数参数。狭窄算法参数由 `DetectorConfig` 管理，CPR 参数由 `CPRConfig` 管理。摘要开关、检测可视化开关和三维标记半径是明确的工作流级关键字参数。综合接口始终保存检测 JSON 和 CPR 图像；`save_detection_visualizations` 只控制额外的检测可视化。

`CombinedResult` 定义为：

```python
@dataclass
class CombinedResult:
    detection: DetectionResult
    cpr_results: list[CPRResult]
    detection_output_dir: str
    cpr_output_dir: str
    warnings: list[str]
```

## 6. CPR 配置

新增 `CPRConfig`：

```python
@dataclass
class CPRConfig:
    angles_degrees: tuple[float, ...] = (0.0,)
    mode: str = "straightened"
    curve_resolution_mm: float = 1.0
    slice_resolution_mm: float = 1.0
    sampling_line_length_mm: float = 40.0
    interpolation_order: int = 1
    analyze_side_branches: bool = True
    centerline_smoothing_sigma_mm: float = 0.8
    min_branch_length_mm: float = 3.0
    min_branch_mean_radius_mm: float = 0.4
    max_branch_order: int | None = None
```

这些默认值与当前 `run_cpr` 行为一致。`run_cpr` 继续接受现有直接参数；其内部可以将经过校验的参数归一化为 `CPRConfig`，但不要求调用者改用配置对象。`run_combined` 使用 `CPRConfig`，避免复制二十多个 CPR 参数。

## 7. 数据流

### 7.1 狭窄检测

```text
读取 label/image/aorta mask
  -> 检查网格和物理几何
  -> 提取中心线树及平行运输框架
  -> 横截面区域生长和质量控制
  -> 计算直径、参考值、狭窄率及管腔平均强度
  -> 合并候选区间
  -> 保存 stenosis_result.json
  -> 可选保存检测可视化
  -> 返回 DetectionResult
```

### 7.2 独立 CPR

```text
读取 image 和可选 label/aorta mask
  -> 使用显式中心线或从标签提取中心线树
  -> 按指定模式和角度生成 CPR
  -> 可选读取 stenosis_result.json
  -> 按完整 branch_id 和中心线距离绘制标记
  -> 可选保存图像
  -> 返回 list[CPRResult]
```

### 7.3 综合流程

```text
一次读取 label/image/aorta mask
  -> 运行狭窄检测并得到 DetectionResult
  -> 保存 JSON 和可选检测可视化
  -> 从 DetectionResult 直接取得分支中心线、距离和框架
  -> 按 CPRConfig 生成 CPR
  -> 使用同一内存结果中的候选绘制狭窄标记
  -> 保存 CPR 图像
  -> 返回 CombinedResult
```

综合流程不得把检测结果先写成 JSON 再读取，也不得再次调用标签中心线树提取。这样保证检测候选和 CPR 标记使用完全相同的分支、点序和距离坐标。

## 8. 输出规则

1. `run_detection` 在用户指定目录写入 `stenosis_result.json` 和启用的检测可视化。
2. `run_cpr` 在用户指定目录写入启用的 CPR 图像。
3. `run_combined` 分别使用 `detection_output_dir` 和 `cpr_output_dir`。
4. 指定目录不存在时，程序创建该目录。
5. 程序不自动增加 `detection`、`cpr` 或其他子目录。
6. 检测目录和 CPR 目录允许指向同一目录。
7. 文件继续使用当前稳定的解剖分支和角度命名规则。
8. 不再写入 `diameter_profiles.csv`、`centerline_branches.csv` 或 `stenosis_candidates.csv`。
9. 独立 CPR 的 JSON 联动保留，因为 `stenosis_result.json` 是跨调用传递候选信息的正式持久化格式。
10. 综合流程虽然仍保存检测 JSON，但 CPR 标记直接使用内存对象。

## 9. 错误处理

### 9.1 立即失败

以下情况抛出明确异常：

- 输入文件不存在或无法读取。
- label、image 或 aorta mask 的尺寸、spacing、origin 或 direction 不兼容。
- CPR 模式、角度、采样分辨率或采样线长度非法。
- 同时提供或同时缺少 `run_cpr` 的两个中心线来源。
- 输出目录无法创建或结果无法写入。

工作流不得吞掉这些错误或返回看似成功的不完整结果。

### 9.2 局部失败

单条血管或分支无法提取中心线、长度不足、平均半径不足或质量控制不通过时，记录警告并跳过对应对象，其余对象继续处理。

`DetectionResult` 保留检测阶段警告。`CombinedResult.warnings` 汇总综合编排和 CPR 阶段的附加警告，例如某个检测分支没有可生成的 CPR。输出文件写入失败仍是致命错误，不降级为警告。

## 10. 兼容性

本次重组有意移除以下旧导入：

```python
from NarrowDetecter.cli import run_detection
from NarrowDetecter.cpr_cli import run_cpr
```

调用方必须改为：

```python
from NarrowDetecter import run_detection, run_cpr, run_combined
```

除导入路径和 CSV 输出外，`run_detection`、`run_cpr` 的现有参数和返回行为保持不变。README 需要同步更新所有示例、输出文件清单和模块说明。

## 11. 验证策略

由于包内不保留测试文件，测试驱动和回归验证使用 `NarrowDetecter` 目录之外的临时测试位置。验证完成后不向包中加入测试目录。

验证至少覆盖：

1. 包根目录能够导入三个工作流函数以及配置、结果类。
2. 旧 `cli.py`、`cpr_cli.py` 删除后不存在残余内部引用。
3. 用户指定目录不存在时可创建，且不产生额外子目录。
4. 狭窄检测保存 JSON 和启用的可视化，但不生成 CSV。
5. 重构前后相同输入的中心线、解剖名称、横截面测量、候选和信号强度指标保持一致。
6. 重构前后 straightened/stretched CPR 数组、坐标轴和背景填充值保持一致。
7. 多角度、主干模式、侧支模式和显式中心线模式保持有效。
8. 独立 CPR 可以从 `stenosis_result.json` 绘制正确分支的狭窄标记。
9. 综合流程只进行一次中心线树提取，并用同一分支对象生成 CPR 和标记。
10. straightened 和 stretched CPR 的标记位置、方向、长度和百分比文本保持现有行为。
11. 输入错误和输出写入错误按设计抛出异常；局部分支失败按设计生成警告。

## 12. 完成标准

满足以下条件后视为重组完成：

- 目标目录结构建立，根目录不再保留旧业务实现文件和旧 CLI 入口。
- 三个正式工作流可以从包根目录单独调用。
- 综合工作流复用检测阶段中心线，不发生重复提取。
- 所有既有核心数值行为和可视化行为通过回归验证。
- 输出目录规则符合本设计，检测流程不再产生 CSV。
- README 与新接口、目录和输出保持一致。
- 临时验证文件不留在 `NarrowDetecter` 包中。
