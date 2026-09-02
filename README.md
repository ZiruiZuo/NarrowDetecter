# NarrowDetecter

`NarrowDetecter` 用于从冠状动脉标签和归一化强度图像中提取中心线树、估计管腔直径、筛查疑似狭窄，并可独立生成 straightened CPR 或 stretched CPR 图像。

当前模块以 Python 函数作为调用接口，不使用命令行参数解析。核心入口是：

- `NarrowDetecter.cli.run_detection`：完整狭窄检测与结果保存。
- `NarrowDetecter.run_cpr`：独立 CPR 生成，可直接从标签提取中心线，也可传入已有中心线。
- `NarrowDetecter.detect_coronary_stenosis`：直接使用 NumPy 数组执行狭窄检测。

> 本模块用于算法研究和辅助分析，输出不能替代临床诊断。

## 1. 目录结构

| 文件 | 作用 |
| --- | --- |
| `cli.py` | 文件级狭窄检测入口 `run_detection` |
| `cpr_cli.py` | 文件级 CPR 入口 `run_cpr` |
| `detector.py` | 横截面测量、参考直径和狭窄候选分析 |
| `centerline.py` | 中心线骨架、根节点、主干和侧支提取 |
| `cross_section.py` | 平行运输框架、横截面采样和区域生长 |
| `cpr.py` | straightened / stretched CPR 的数组级实现 |
| `anatomy.py` | 分支解剖名称分配 |
| `visualization.py` | 直径曲线、候选平面、标记体数据和 CPR 图像 |
| `models.py` | 配置和结果数据类 |
| `myio.py` | SimpleITK 输入输出与空间信息校验 |

## 2. 环境安装

建议使用 Python 3.10 或更高版本。

```powershell
cd E:\ZZR\Code\Tools
pip install -r NarrowDetecter\requirements.txt
```

依赖包括 `numpy`、`scipy`、`scikit-image`、`SimpleITK` 和 `matplotlib`。

当前目录没有独立的安装包配置。调用时请从 `E:\ZZR\Code\Tools` 启动 Python，或将该目录加入 `PYTHONPATH`。

## 3. 输入文件

### 3.1 必需文件

狭窄检测需要两个三维医学图像：

1. **冠脉标签 `label_path`**
   - 三维整数标签图像。
   - 默认标签值为 `LAD=2`、`LCX=3`、`RCA=4`。
   - 可通过 `DetectorConfig.label_values` 自定义。

2. **归一化强度图像 `image_path`**
   - 三维浮点图像。
   - 应与标签处于同一物理空间。
   - 期望强度大致位于 `[0, 1]`；超出 `[-0.05, 1.05]` 会产生警告，但不会直接终止。

文件格式由 SimpleITK 支持，例如 `.nii`、`.nii.gz`、`.mha`、`.mhd`、`.nrrd`。

### 3.2 可选主动脉掩膜

`aorta_mask_path` 用于判断冠脉树的近端 root：

- 仅体素值等于 `1` 的区域被视为主动脉。
- 掩膜必须与强度图像处于同一尺寸和物理空间。
- 有主动脉掩膜时，优先选择最靠近主动脉的骨架端点作为 root。
- 未提供时，使用局部血管半径最大的骨架端点作为 root。

### 3.3 空间一致性

文件级接口会检查标签、强度图像和主动脉掩膜的：

- `size`
- `spacing`
- `origin`
- `direction`

它们必须一致。NumPy 数组和内部坐标均采用 `(z, y, x)` 顺序；SimpleITK 的 spacing 原本为 `(x, y, z)`，读取后会转换为 `(z, y, x)`。

## 4. 快速开始：狭窄检测

```python
from NarrowDetecter import DetectorConfig
from NarrowDetecter.cli import run_detection

config = DetectorConfig(
    label_values={"LAD": 2, "LCX": 3, "RCA": 4},
    analyze_side_branches=True,
    candidate_threshold=0.25,
)

result = run_detection(
    label_path=r"E:\data\coronary_label.nii.gz",
    image_path=r"E:\data\image_normalized.nii.gz",
    output_dir=r"E:\data\NarrowResult",
    aorta_mask_path=r"E:\data\aorta_mask.nii.gz",  # 可省略
    config=config,
    print_summary=True,
    save_visualizations=True,
)
```

`run_detection` 返回 `DetectionResult`，并按配置将表格、JSON 和可视化结果写入 `output_dir`。

### 4.1 `run_detection` 参数

```python
run_detection(
    label_path,
    image_path,
    output_dir,
    config=None,
    print_summary=True,
    save_visualizations=True,
    marker_radius_mm=1.5,
    aorta_mask_path=None,
    centerline_label_radius_mm=0.6,
)
```

| 参数 | 说明 |
| --- | --- |
| `label_path` | 冠脉多分类标签路径 |
| `image_path` | 归一化三维强度图像路径 |
| `output_dir` | 输出目录 |
| `config` | `DetectorConfig`；`None` 时使用默认值 |
| `print_summary` | 是否打印各血管、分支和候选摘要 |
| `save_visualizations` | 是否生成曲线、候选平面和 NIfTI 标记图 |
| `marker_radius_mm` | 疑似狭窄三维标记球半径 |
| `aorta_mask_path` | 可选主动脉掩膜路径 |
| `centerline_label_radius_mm` | 解剖名称中心线标签图的绘制半径 |

### 4.2 数组级调用

已经在内存中持有数据时，可直接调用：

```python
import SimpleITK as sitk
from NarrowDetecter import DetectorConfig, detect_coronary_stenosis

label_itk = sitk.ReadImage(r"E:\data\coronary_label.nii.gz")
image_itk = sitk.ReadImage(r"E:\data\image_normalized.nii.gz")

label = sitk.GetArrayFromImage(label_itk)
image = sitk.GetArrayFromImage(image_itk)
spacing_zyx = image_itk.GetSpacing()[::-1]

aorta_itk = sitk.ReadImage(r"E:\data\aorta_mask.nii.gz")
aorta_mask = sitk.GetArrayFromImage(aorta_itk) == 1

result = detect_coronary_stenosis(
    label=label,
    image=image,
    spacing_zyx=spacing_zyx,
    config=DetectorConfig(),
    aorta_mask=aorta_mask,
)
```

数组级接口不会自动校验 SimpleITK 的 `origin` 和 `direction`，调用方需要保证数组已经对齐。

## 5. 狭窄检测流程

每个冠脉类别单独执行以下步骤：

1. 保留对应标签的最大连通区域并骨架化。
2. 将 26 邻域骨架构造成加权图。
3. 确定 root，并取 root 到各末端的最短路径并集形成中心线树。
4. 主干编号 `B000`，取 root 到最远末端的完整路径；其余拓扑段作为侧支。
5. 按弧长重采样和平滑每条分支，并建立平行运输框架 `(u_i, v_i)`。
6. 在每个中心线点建立垂直于切向量的横截面。
7. 在横截面内，以中心点附近为 seed，联合强度和梯度进行受标签约束的区域生长。
8. 计算管腔面积和等价圆直径 `d = 2 * sqrt(area / pi)`。
9. 建立平滑直径曲线和局部参考直径，计算狭窄率并合并连续候选区间。

当前分支检测是基于骨架拓扑的几何启发式方法，并不是严格的冠脉解剖识别模型。

## 6. 参考直径与狭窄率

对中心线位置 `i`，程序在以该位置为中心的局部窗口内选择可靠测量值：

- 默认窗口长度：`reference_window_mm=20.0`。
- 排除目标点附近 `reference_exclusion_mm=1.5` 的区域。
- 优先排除末端区和分叉区。
- 只使用 `quality > 0.2` 的测量。
- 取候选直径的 `reference_percentile=80` 百分位作为参考直径。

直径狭窄率：

```text
diameter_stenosis = 1 - measured_diameter / reference_diameter
```

面积狭窄率：

```text
area_stenosis = 1 - measured_area / reference_area
```

默认将直径狭窄率达到 `25%` 且连续长度至少 `1 mm` 的区域作为候选，相邻间隔不超过 `1 mm` 的候选会被合并。

分级规则：

| 最大直径狭窄率 | `grade` |
| --- | --- |
| `< 25%` | `minimal` |
| `25% - <50%` | `mild` |
| `50% - <70%` | `moderate` |
| `70% - <100%` | `severe` |
| `>= 100%` | `occlusion_suspected` |

## 7. 区域生长质量与标签回退

区域生长结果出现以下任一情况时被视为无效：

- 面积小于 `min_lumen_area_mm2`。
- 相对标签先验横截面面积小于 `min_area_ratio_to_prior`。
- 相对标签先验横截面面积大于 `max_area_ratio_to_prior`。
- 生长区域没有包含中心 seed。

当 `use_label_fallback=True` 且标签先验区域有效时，程序改用原标签在该横截面上的连通区域，测量 flags 中会包含 `label_fallback`。

质量分数主要由中心包含程度决定，并会受到以下惩罚：

- 标签回退结果的质量上限为 `0.45`。
- 生长区域接触采样窗口边界时乘以 `0.6`。
- 靠近血管末端时乘以 `0.5`。
- 靠近分叉点时乘以 `0.5`。

当前代码以 `quality <= 0.2` 作为低质量测量：它不参与可靠参考直径选择，也不会用于生成狭窄候选。

## 8. `DetectorConfig`

### 中心线与分支

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `label_values` | `LAD=2, LCX=3, RCA=4` | 冠脉名称到标签值的映射 |
| `centerline_step_mm` | `0.4` | 中心线重采样间距 |
| `centerline_smoothing_sigma_mm` | `0.8` | 中心线平滑尺度 |
| `analyze_side_branches` | `True` | 是否保留并分析侧支 |
| `min_branch_length_mm` | `3.0` | 最短保留分支长度 |
| `min_branch_mean_radius_mm` | `0.4` | 最小平均半径 |
| `max_branch_order` | `None` | 最大分支级数；`None` 不限制 |

### 横截面与区域生长

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `cross_section_pixel_mm` | `0.15` | 横截面像素间距 |
| `cross_section_radius_mm` | `5.0` | 横截面半径，完整边长约为 `10 mm` |
| `mask_dilation_mm` | `1.2` | 标签约束区域的膨胀距离 |
| `image_smoothing_sigma_mm` | `0.25` | 强度图预平滑尺度 |
| `seed_radius_mm` | `0.45` | 中心 seed 半径 |
| `intensity_tolerance` | `0.20` | 相对 seed 强度容差 |
| `gradient_percentile` | `82.0` | 梯度归一化百分位 |
| `intensity_weight` | `0.55` | 区域代价中的强度权重 |
| `gradient_weight` | `0.45` | 区域代价中的梯度权重 |
| `region_cost_threshold` | `0.62` | 区域生长代价阈值 |

### 面积、参考值和候选

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `min_lumen_area_mm2` | `0.20` | 最小有效管腔面积 |
| `min_area_ratio_to_prior` | `0.08` | 相对标签先验的最小面积比 |
| `max_area_ratio_to_prior` | `1.8` | 相对标签先验的最大面积比 |
| `reference_window_mm` | `20.0` | 局部参考窗口长度 |
| `reference_exclusion_mm` | `1.5` | 目标点附近排除长度 |
| `reference_percentile` | `80.0` | 参考直径百分位 |
| `profile_smoothing_mm` | `1.0` | 直径曲线平滑尺度 |
| `candidate_threshold` | `0.25` | 候选直径狭窄率阈值 |
| `min_stenosis_length_mm` | `1.0` | 最短候选长度 |
| `merge_gap_mm` | `1.0` | 候选合并的最大间隔 |
| `endpoint_exclusion_mm` | `2.0` | 末端低置信区长度 |
| `bifurcation_exclusion_mm` | `1.5` | 分叉低置信区长度 |
| `use_label_fallback` | `True` | 区域生长失败时是否使用标签先验 |

## 9. 分支和解剖名称

中心线树先按拓扑切分，再按冠脉类别分配名称：

- **LAD**：root 到第一个主干分叉点为 `LM`，后续最长主路径为 `LAD`；其他侧支按近端到远端命名为 `D_1`、`D_2` 等。
- **LCX**：最长主路径为 `LCX`；其他侧支命名为 `OM_1`、`OM_2` 等。
- **RCA**：主路径近端为 `RCA`、远端为 `PDA`；第一条侧支为 `PLB`，后续为 `PLB_2`、`PLB_3` 等。

若 LAD 或 RCA 没有找到适合拆分的主干分叉点，则主支分别保持为 `LAD` 或 `RCA`。名称用于表格、图标题和输出文件名；结果中的稳定分支 ID 使用 `LAD-B000`、`LAD-B001` 这类“血管名 + 分支编号”的形式。

## 10. 狭窄检测输出

`run_detection` 总会写出：

- `stenosis_result.json`：完整结构化结果。

存在相应数据时还会写出：

- `diameter_profiles.csv`：每个中心线点的面积、直径、参考值、狭窄率、质量和 flags。
- `centerline_branches.csv`：分支拓扑、长度、半径和解剖名称。
- `stenosis_candidates.csv`：疑似狭窄区间及分级。

当 `save_visualizations=True` 时，还会生成：

- `*_diameter_profile.png`：血管直径、参考直径和狭窄率曲线。
- `*_candidate_XX_cross_section.png`：候选点处垂直于中心线的平面，包含区域生长分割轮廓和等价圆轮廓。
- `stenosis_markers.nii.gz`：疑似狭窄位置的三维球形标记，标记值沿用相应冠脉标签值。
- `centerline_anatomy_labels.nii.gz`：带解剖名称编号的三维中心线标签图。
- `centerline_anatomy_labels.json`：解剖名称与标签值的对应关系。

## 11. 独立生成 CPR

CPR 不需要和狭窄检测一起运行。`run_cpr` 支持两种互斥输入方式：

1. 提供 `label_path`，函数内部提取中心线树。
2. 提供 `centerline_voxel_zyx`，直接使用已有中心线。

两者必须且只能提供一个。

### 11.1 从冠脉标签生成 CPR

```python
from NarrowDetecter import run_cpr

results = run_cpr(
    image_path=r"E:\data\image_normalized.nii.gz",
    label_path=r"E:\data\coronary_label.nii.gz",
    aorta_mask_path=r"E:\data\aorta_mask.nii.gz",  # 可省略
    output_dir=r"E:\data\CPR",
    label_values={"LAD": 2, "LCX": 3, "RCA": 4},
    angles_degrees=(0.0, 45.0, 90.0),
    mode="straightened",
    curve_resolution_mm=1.0,
    slice_resolution_mm=1.0,
    sampling_line_length_mm=40.0,
)
```

### 11.2 从已有中心线生成 CPR

```python
import numpy as np
from NarrowDetecter import run_cpr

centerline_voxel_zyx = np.load(r"E:\data\centerline.npy")

results = run_cpr(
    image_path=r"E:\data\image_normalized.nii.gz",
    centerline_voxel_zyx=centerline_voxel_zyx,
    output_dir=r"E:\data\CPR",
    angles_degrees=(0.0, 45.0),
    mode="stretched",
    curve_resolution_mm=1.0,
    slice_resolution_mm=1.0,
    sampling_line_length_mm=40.0,
    name="LAD",
)
```

`centerline_voxel_zyx` 的形状必须为 `(N, 3)`，坐标顺序为 `(z, y, x)`。默认会按 `curve_resolution_mm` 重采样并按 `centerline_smoothing_sigma_mm` 平滑，再计算框架或 stretched 几何。

### 11.3 `run_cpr` 参数

```python
run_cpr(
    image_path,
    centerline_voxel_zyx=None,
    output_dir=None,
    angles_degrees=(0.0,),
    mode="straightened",
    curve_resolution_mm=1.0,
    slice_resolution_mm=1.0,
    sampling_line_length_mm=40.0,
    frames_u_zyx=None,
    frames_v_zyx=None,
    centerline_distances_mm=None,
    distance_from_root_mm=0.0,
    interpolation_order=1,
    name="centerline",
    save_outputs=True,
    print_summary=True,
    label_path=None,
    label_values=None,
    aorta_mask_path=None,
    analyze_side_branches=True,
    centerline_smoothing_sigma_mm=0.8,
    min_branch_length_mm=3.0,
    min_branch_mean_radius_mm=0.4,
    max_branch_order=None,
)
```

主要采样参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `mode` | `straightened` | `straightened` 或 `stretched` |
| `angles_degrees` | `(0.0,)` | CPR 观察角度，可一次生成多个角度 |
| `curve_resolution_mm` | `1.0` | 沿中心线的重采样间距 |
| `slice_resolution_mm` | `1.0` | 沿采样线及 stretched 画布的像素间距 |
| `sampling_line_length_mm` | `40.0` | 以中心线点为中心的采样线长度 |
| `interpolation_order` | `1` | `scipy.ndimage.map_coordinates` 插值阶数，范围 `0-5` |
| `save_outputs` | `True` | 是否保存 PNG；为 `False` 时可不提供 `output_dir` |

`sampling_line_length_mm` 是 CPR 应暴露给交互界面的采样宽度。狭窄检测中的 `cross_section_radius_mm` 则控制二维横截面分析窗口，两者互不影响。

传入已有中心线时，还可同时提供匹配的 `frames_u_zyx`、`frames_v_zyx` 和 `centerline_distances_mm`。`frames_u_zyx` 与 `frames_v_zyx` 必须成对提供；若提供显式框架或距离数组，`run_cpr` 不再自动重采样这条中心线。

## 12. Straightened CPR

中心线每个点 `P_i` 有平行运输得到的横截面基向量 `u_i` 和 `v_i`。角度 `phi` 对应的采样方向为：

```text
l_i(phi) = cos(phi) * u_i + sin(phi) * v_i
```

每个中心线点沿 `l_i` 采样一条长度为 `sampling_line_length_mm` 的直线，再按中心线顺序拼成二维图像：

- 横轴：沿中心线的累计弧长。
- 纵轴：沿采样线的有符号偏移。
- 图中 `y=0`：三维中心线。

这里的 `(u_i, v_i)` 位于垂直于该点中心线切向量的平面中。它们由第一点的初始横截面基向量开始，通过平行运输沿整条中心线传播，以减少逐点坐标系突然翻转。

## 13. Stretched CPR

Stretched CPR 使用患者物理坐标中的全局 vector-of-interest。当前约定：

- SimpleITK 物理坐标采用 LPS `(x, y, z)`。
- 全局 `+z` 为患者轴向方向，`xy` 平面为横断面。
- `phi=0` 对应 LPS `+x`，角度增大时在全局 `xy` 平面旋转。
- 全局采样方向为 `(cos(phi), sin(phi), 0)`，再通过图像 direction 矩阵转换到图像数组的 `(z, y, x)` 坐标。

对相邻物理中心线点 `P_i`、`P_(i+1)`，令：

```text
d_i = P_(i+1) - P_i
dx_i = dot(l, d_i)
dy_i = sqrt(max(||d_i||^2 - dx_i^2, 0))
```

二维中心线通过累加得到：

```text
x_(i+1) = x_i + dx_i
y_(i+1) = y_i + dy_i
```

因此相邻中心线点在展开图上的距离仍等于三维物理距离。主干从 root 开始，首点定义为 `(0, 0)`；侧支结果以该侧支自己的起点作为局部起点。

最终图像不是用二维中心线点重新寻找三维位置，而是保持 `P_i` 与二维位置 `Q_i` 的既定对应。画布上每一行沿同一个全局 vector-of-interest 回到原始三维图像采样，并在相邻采样行之间插值形成连续图像。

### Stretched 画布宽度

stretched 模式的横向画布需要容纳二维中心线在 vector-of-interest 方向上的累计位移，因此可能宽于 `sampling_line_length_mm`。当前实现保留这一行为：每一行按完整画布宽度采样，不会将超出指定采样线长度的部分裁掉或填充为背景。

## 14. CPR 输出

`run_cpr` 返回 `list[CPRResult]`。默认只保存 PNG，不保存 `.npz`。

文件名示例：

- `LAD_CPR_phi_45deg.png`
- `LAD_stretched_CPR_phi_45deg.png`

`CPRResult` 的主要字段：

| 字段 | 含义 |
| --- | --- |
| `image` | 二维 CPR 强度数组 |
| `centerline_distances_mm` | 原三维中心线累计弧长 |
| `distances_from_root_mm` | 每点距整棵树 root 的距离 |
| `offsets_mm` | straightened 模式中的采样线偏移 |
| `angle_degrees` | CPR 角度 |
| `branch_id` | 内部分支 ID |
| `anatomical_label` | 解剖名称 |
| `point_anatomical_labels` | 每个中心线点的解剖名称 |
| `mode` | `straightened` 或 `stretched` |
| `display_x_mm`, `display_y_mm` | stretched 画布坐标 |
| `centerline_x_mm`, `centerline_y_mm` | stretched 二维中心线坐标 |
| `vector_of_interest_zyx` | 图像数组坐标中的全局采样方向 |
| `vector_of_interest_lps_xyz` | 患者 LPS 坐标中的全局采样方向 |
| `sampling_line_length_mm` | 请求的采样线长度 |

stretched JPG 中：

- `coordinate along vector of interest` 对应横向 `display_x_mm`。
- `accumulated perpendicular distance` 对应纵向 `display_y_mm`，由每段中心线位移中垂直于 vector-of-interest 的分量逐段累加得到。

## 15. 其他公开 API

包根目录还导出以下数组级函数和数据类：

- `extract_centerline_tree`
- `sample_cpr`
- `generate_centerline_cpr`
- `generate_stretched_cpr`
- `generate_label_cprs`
- `generate_branch_cpr`
- `global_vector_of_interest_zyx`
- `DetectorConfig`
- `DetectionResult`
- `BranchResult`
- `StenosisCandidate`
- `CPRResult`

通常优先使用 `run_detection` 和 `run_cpr`；只有在需要嵌入其他处理管线、复用已有数组或自定义保存逻辑时，才直接使用这些底层接口。

## 16. 常见问题

### 中心线方向看起来反了怎么办？

优先提供 `aorta_mask_path`。未提供主动脉掩膜时，root 依赖“端点局部半径最大”的启发式规则，在近远端直径接近或标签不完整时可能判断错误。

### CPR 边界模糊是什么原因？

中心线已经经过按物理弧长重采样和平滑，三维图像也采用插值采样。模糊程度还会受到原图分辨率、`curve_resolution_mm`、`slice_resolution_mm`、插值阶数、中心线偏移和部分容积效应影响。

### 为什么有些位置使用原标签而不是区域生长？

区域生长面积异常、没有包含 seed 或低于最小面积时会触发标签回退。可在 `diameter_profiles.csv` 的 `flags` 和 `quality` 中检查具体位置。

### 为什么某些分支名称和预期解剖不完全一致？

名称来自中心线拓扑、长度和近远端顺序的启发式映射。复杂分叉、断裂标签或缺失侧支可能影响命名，最终结果应结合三维图像核对。
