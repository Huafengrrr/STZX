# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def set_font(run, name='Microsoft YaHei', size=11, bold=False, color=None, italic=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    if italic is not None:
        run.font.italic = italic

def add_heading_custom(doc, text, level=1, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_font(run, size=18 if level==1 else 14, bold=True, color=color)
    if level == 1:
        p.add_run().add_break()
    return p

def add_qa(doc, question, answer_lines):
    # Question
    p = doc.add_paragraph()
    from docx.oxml import OxmlElement
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), 'F0F4FA')
    p._element.get_or_add_pPr().append(shd)
    r = p.add_run(question)
    set_font(r, size=11.5, bold=True, color=(196,89,17))
    
    for line in answer_lines:
        if line.startswith('- '):
            p = doc.add_paragraph(line[2:], style='List Bullet')
            for run in p.runs:
                set_font(run, size=10.5)
        else:
            p = doc.add_paragraph(line)
            for run in p.runs:
                set_font(run, size=10.5)

doc = Document()

# === 页面设置 ===
section = doc.sections[0]
section.page_width = Cm(21)
section.page_height = Cm(29.7)
section.top_margin = Cm(2.5)
section.bottom_margin = Cm(2.5)
section.left_margin = Cm(2.2)
section.right_margin = Cm(2.2)

# === 封面 ===
for _ in range(4):
    doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("声途智行")
set_font(r, size=28, bold=True, color=(46,117,182))

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("智能盲杖 Android 辅助导航应用")
set_font(r, size=14, color=(102,102,102))

for _ in range(3):
    doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("全国大学生计算机设计大赛")
set_font(r, size=15)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("答 辩 应 对 材 料")
set_font(r, size=17, bold=True)

# ===== 一、项目总体介绍 =====
doc.add_page_break()
h = doc.add_heading("一、项目总体介绍（开场陈述）", level=1)
for r in h.runs:
    set_font(r, size=16, bold=True, color=(46,117,182))

p = doc.add_paragraph()
r1 = p.add_run('"声途智行"')
set_font(r1, size=12, bold=True, color=(46,117,182))
r2 = p.add_run(' 是一款面向视障人士的智能导航辅助 Android 应用，配合智能盲杖硬件使用。核心理念是让视障用户\u201c用耳朵看世界\u201d\u2014\u2014通过 AI 视觉感知周围障碍、用语音实时播报位置与导航、并在紧急情况下自动向家属发出求救信号。')
set_font(r2, size=11)

doc.add_heading("五大核心功能", level=2)

table_data = [
    ("功能模块", "技术实现方案"),
    ("障碍物识别 + 测距", "YOLOv8 TFLite 7类目标检测 + OpenCV SGBM 双目立体测距"),
    ("实时语音导航", "高德步行路线规划 API + Android TTS 逐步播报"),
    ("位置感知 + 播报", "GPS + 高德逆地理编码 + 自动街道变更播报"),
    ("紧急求救（SOS + 跌倒）", "加速度传感器双阶段跌倒检测 + 音量键 SOS + uniCloud 位置上报"),
    ("多模态 AI 问答", 'Qwen3-VL:8b 多模态大语言模型，支持\u201c看图回答\u201d语音问询'),
]

t = doc.add_table(rows=len(table_data), cols=2)
t.style = 'Table Grid'
t.autofit = True
for i, (col1, col2) in enumerate(table_data):
    row = t.rows[i]
    c0, c1 = row.cells[0], row.cells[1]
    if i == 0:
        for c in [c0, c1]:
            shading = OxmlElement('w:shd')
            shading.set(qn('w:fill'), '2E75B6')
            c._tc.get_or_add_tcPr().append(shading)
    elif i % 2 == 0:
        for c in [c0, c1]:
            shading = OxmlElement('w:shd')
            shading.set(qn('w:fill'), 'F5F8FC')
            c._tc.get_or_add_tcPr().append(shading)
    r0 = c0.paragraphs[0].clear()
    r1 = c1.paragraphs[0].clear()
    run0 = r0.add_run(col1); set_font(run0, size=10.5, bold=i==0, color=(255,255,255) if i==0 else None)
    run1 = r1.add_run(col2); set_font(run1, size=10.5, color=(255,255,255) if i==0 else None)

# ===== 二、系统架构 =====
doc.add_page_break()
h = doc.add_heading("二、系统架构说明", level=1)
for r in h.runs: set_font(r, size=16, bold=True, color=(46,117,182))

p = doc.add_paragraph("本应用采用分层架构设计，自底向上分为四个层次：")

arch_data = [
    ("层次", "组件与职责"),
    ("硬件层", "USB UVC 双目摄像头（1280×480）、Android GPS 传感器、三轴加速度传感器"),
    ("感知层\n（本地AI推理）", "StereoVision.kt — OpenCV SGBM 立体视觉测距（0.2~12m）\nCameraFrameProcessor.kt — YOLOv8 TFLite 目标检测（3~5 FPS）\nFallDetector.kt — 失重+撞击双阶段跌倒检测"),
    ("决策与交互层", "LocationHelper.kt — GPS定位+高德逆地理编码+位置上传\nNavigationManager.kt — 步行路线规划+逐步播报\nMainActivity.kt — 语音识别+TTS播报+SOS求救\nDeviceBindingManager.kt — 设备配对与家属端绑定"),
    ("云端层", "uniCloud 云函数（位置数据存储与推送）\nAutoDL Qwen3-VL:8b（多模态AI大模型）\n高德 Web API（逆地理编码+步行导航）"),
]

t2 = doc.add_table(rows=len(arch_data), cols=2)
t2.style = 'Table Grid'
for i, (col1, col2) in enumerate(arch_data):
    row = t2.rows[i]
    c0, c1 = row.cells[0], row.cells[1]
    if i == 0:
        for c in [c0, c1]:
            shading = OxmlElement('w:shd'); shading.set(qn('w:fill'), '2E75B6')
            c._tc.get_or_add_tcPr().append(shading)
    elif i % 2 == 0:
        for c in [c0, c1]:
            shading = OxmlElement('w:shd'); shading.set(qn('w:fill'), 'F5F8FC')
            c._tc.get_or_add_tcPr().append(shading)
    r0 = c0.paragraphs[0].clear(); run0 = r0.add_run(col1)
    set_font(run0, size=10, bold=i==0, color=(255,255,255) if i==0 else None)
    r1 = c1.paragraphs[0].clear(); run1 = r1.add_run(col2.replace('\n', '\n'))
    set_font(run1, size=10, color=(255,255,255) if i==0 else None)

# ===== 三、核心技术问答 =====
doc.add_page_break()
h = doc.add_heading("三、核心技术问题应对", level=1)
for r in h.runs: set_font(r, size=16, bold=True, color=(46,117,182))

qa_list = [
    (
        "Q1：目标检测用什么模型？识别准确率如何？",
        [
            "我们使用 YOLOv8 经过 TFLite 量化后的版本，专门针对视障出行场景训练，识别 7 个类别：",
            "- 人行道、行人、柱子、汽车、卡车、公交车、自行车",
            "其中行人因为形态多样，置信度阈值特别调低到 0.35，其余类别为 0.5，在减少漏检的同时控制误报。",
            "在 ARM64 手机上运行速度约 3~5 FPS，对于辅助导航场景已经足够——用户步行速度较慢，不需要像自动驾驶那样高帧率。",
            "代码对应位置：CameraFrameProcessor.kt 第113行的阈值设定。",
        ]
    ),
    (
        "Q2：距离怎么测的？精度如何？",
        [
            "我们采用双目立体视觉测距，使用 OpenCV 的 SGBM（半全局块匹配）算法生成视差图，再用标定参数将视差值换算成真实距离。",
            "核心公式：距离 = (焦距 × 基线长度) / 视差",
            "具体参数：",
            "- 焦距：约328像素（实际标定值）",
            "- 基线长度：约61mm（两摄像头间距）",
            "- 测距范围：0.2m ~ 12m",
            "两层优化提高鲁棒性：",
            "- MAD（中位绝对偏差）异常值剔除：剔除视差图中的噪声点，再取四分位均值",
            "- EMA 指数平滑：当距离突变超过1.5m时平滑系数降为0.1，正常情况为0.3，避免测距跳变",
            "代码对应位置：CameraFrameProcessor.kt 第224~249行。",
        ]
    ),
    (
        "Q3：同一物体在连续帧之间怎么保持稳定？",
        [
            "我们实现了基于 IoU（交并比）的目标追踪 + EMA 平滑。",
            "每一帧新检测到的目标，都会和上一帧的目标逐一计算 IoU 重叠度，如果 IoU > 0.5 就认为是同一个目标，然后用 EMA 公式对位置和大小做平滑：",
            "  新位置 = α × 当前检测值 + (1-α) × 上帧平滑值   （α = 0.3）",
            "这样就算某帧检测抖动了，显示的检测框也不会突然跳位，非常稳定。",
        ]
    ),
    (
        "Q4：双目摄像头的标定参数是怎么来的？",
        [
            "通过 OpenCV 的双目标定流程，用棋盘格标定板从不同角度拍摄若干张图像，由 OpenCV 自动求解出：",
            "- 左右相机的内参矩阵（焦距、光心）",
            "- 畸变系数（镜头畸变校正）",
            "- 左右相机的相对旋转矩阵 R 和平移向量 T（基线长约61mm）",
            "然后调用 stereoRectify 计算立体校正映射矩阵，运行时对每帧图像做极线对齐，确保 SGBM 能正确计算视差。",
        ]
    ),
    (
        "Q5：语音导航怎么做？能识别用户说话吗？",
        [
            "分两个方向：",
            "【用户输入】用户按音量下键触发 Android 系统语音识别（RecognizerIntent），解析结果后判断意图：",
            '- 包含"导航到/去/带我到"等词 → 解析目的地，发起路线规划',
            '- 包含"在哪/位置/定位"等词 → 主动查询并播报当前位置',
            "- 其他内容 → 发送给 Qwen-VL 大模型做图文问答",
            "【路线播报】调用高德步行导航 Web API（两步：先地理编码得到目的地坐标，再规划路线），返回的每段 Step 保存下来，每15秒 GPS 巡航时检测与下一节点距离（Haversine 公式），进入20m范围内自动播报下一段指令。",
        ]
    ),
    (
        "Q6：SOS 和跌倒检测有什么区别？怎么触发的？",
        [
            "两种机制完全独立：",
            '【主动 SOS（手动触发）】快速按音量上键 3次（2秒内）触发，有5秒防抖冷却防止误触。触发后语音播报，并上传带"SOS"标记的坐标到云端。',
            "【被动跌倒检测（自动触发）】由 FallDetector.kt 监听加速度传感器，采用双阶段检测：",
            "- 先检测失重期（合加速度 < 4 m/s²）",
            '- 失重后1秒内出现撞击峰值（合加速度 > 20 m/s²），才判定为摔倒',
            '触发后自动语音警报，上传带"FALL"标记的坐标。',
        ]
    ),
    (
        "Q7：Qwen-VL 大模型是本地跑还是云端？怎么用的？",
        [
            "部署在 AutoDL 云服务器（A10 GPU），使用 Ollama 框架提供推理服务，Android 端通过 HTTP 请求访问。",
            "处理流程：",
            "1. 截取当前摄像头画面（左眼图，640×480）",
            "2. 压缩为 Base64 JPEG 编码",
            "3. 构建包含图像和用户问题的 JSON 请求体，发给 Qwen3-VL:8b",
            "Prompt 中约束模型：必须在20字以内用一句话回答，不允许废话——视障用户需要简洁快速的回答。",
        ]
    ),
    (
        "Q8：家属端怎么知道视障用户的位置？",
        [
            "通过 uniCloud 云函数作为中间层：",
            "1. 盲杖 App 每15秒上传一次 GPS 坐标（GCJ02火星坐标系），带上设备ID和状态标记（Normal / SOS / FALL）",
            "2. 亲友端 App 通过设备配对码绑定后，定时从云端拉取最新位置数据",
            "3. 发生 SOS 或跌倒时立即上传，不等下一个15秒周期，确保实时性",
            "坐标统一转换：Android GPS 输出 WGS84 标准，高德地图用 GCJ02（火星坐标系），我们在 StereoVision.wgs84ToGcj02() 中做了转换。",
        ]
    ),
    (
        "Q9：SGBM 防撞兜底是什么？",
        [
            'YOLO 只能识别训练过的7类物体。如果前方有没见过的东西（如临时柱子、路障），YOLO不会报警但非常危险。',
            "所以我们增加了 SGBM 全局防撞兜底：",
            "- 当 YOLO 没有触发任何紧急警报时",
            '- 在画面中央 ROI 区域（宽1/4~3/4，高1/2~底部40px）密集采样视差值',
            '- 如果距离 < 1m 的点超过50个 → 直接播报"警告！距离未知物体过近，请避让！"',
            "这样覆盖了 YOLO 的盲区，形成双重保障。",
        ]
    ),
    (
        "Q10：为什么坐标用 GCJ02 而不是 WGS84？",
        [
            '中国国家规定国内商业地图（高德、百度）必须使用 GCJ02（"火星坐标系"）。Android手机GPS输出的是国际标准WGS84坐标，直接传给高德API会出现几十到几百米的系统性偏移。',
            "我们在 StereoVision.wgs84ToGcj02() 实现了该转换算法，所有GPS数据在上传或调用高德API之前都先经过转换，确保地图显示准确。",
        ]
    ),
]

for q, ans_lines in qa_list:
    add_qa(doc, q, ans_lines)

# ===== 四、设计选型 =====
doc.add_page_break()
h = doc.add_heading("四、设计选型相关问题", level=1)
for r in h.runs: set_font(r, size=16, bold=True, color=(46,117,182))

design_qa = [
    (
        "为什么用双目摄像头而不用 ToF 或超声波测距？",
        [
            "三个方面考量：",
            "- 信息量：双目同时提供彩色图像（供YOLO识别）和深度（SGBM测距），一个硬件解决两个问题；ToF只给深度点云，超声波只给一个距离值",
            "- 成本：双目USB摄像头成本百元级别，ToF模块成本较高",
            "- 扩展性：有图像才能接入LLM做图文问答，这是本项目核心特色功能",
            "代价是计算量更大，但通过模型量化+单线程推理+3~5 FPS帧率控制，在手机上可以实时运行。",
        ]
    ),
    (
        "为什么用 TFLite 而不是 ONNX 或 CoreML？",
        [
            "TFLite 是Android平台官方端侧推理框架：",
            "- 原生支持 Android NDK，集成最简单",
            "- 支持 ARM NEON 硬件加速",
            "- INT8量化后模型仅约70MB，适合移动端",
        ]
    ),
    (
        "语音播报优先级怎么控制的？",
        [
            "Android TTS 支持 QUEUE_FLUSH（打断当前立即播报）和 QUEUE_ADD（排队等待播报）两种模式。",
            "策略：",
            '- 优先级高（QUEUE_FLUSH 打断）：紧急避让警报、SOS/跌倒警报、LLM回答、按键触发位置查询',
            '- 优先级低（QUEUE_ADD 排队）：正常障碍物预警、导航步骤播报、街道进入播报',
            "同时设防抖：普通障碍物播报间隔 ≥2500ms，紧急警报间隔 ≥2000ms，避免重复轰炸。",
        ]
    ),
]

for q, ans_lines in design_qa:
    add_qa(doc, q, ans_lines)

# ===== 五、创新点 =====
doc.add_page_break()
h = doc.add_heading("五、创新点总结（答辩重点强调）", level=1)
for r in h.runs: set_font(r, size=16, bold=True, color=(46,117,182))

innovations = [
    "双重保障感知：YOLO 识别已知障碍 + SGBM 全局扫描兜底未知障碍，覆盖所有近距危险",
    "双阶段跌倒检测：失重期 + 撞击期组合判断，精准区分跌倒与普通走路抖动",
    "端云协同 AI：本地轻量模型做实时感知，云端大模型做图文理解，充分利用算力",
    "全语音交互：操作无需看屏幕，音量键控制，语音识别输入，TTS 输出，完全无障碍",
    "家属安全链路：设备绑定 + 实时位置同步 + SOS/跌倒事件推送，构建完整安全保障",
]

for idx, text in enumerate(innovations):
    p = doc.add_paragraph(style='List Number')
    for run in p.runs:
        run.text = text
        set_font(run, size=11)

# ===== 六、挑战性问题 =====
doc.add_page_break()
h = doc.add_heading("六、可能的挑战性问题应对", level=1)
for r in h.runs: set_font(r, size=16, bold=True, color=(46,117,182))

challenge_qa = [
    (
        "室内没有 GPS 怎么办？",
        [
            "设计了 GPS/网络双模式：先尝试GPS，不可用则降级到网络定位（基站/WiFi）。",
            "室内场景目前是设计局限，但产品定位为户外出行辅助，室内导航是未来迭代方向（可接入BLE蓝牙信标或WiFi定位）。",
        ]
    ),
    (
        "网络不好的情况下能正常工作吗？",
        [
            "分级降级设计：",
            "- 位置上传失败自动重试2次（3秒间隔），失败后继续不影响本机功能",
            "- 导航和LLM依赖网络，但 YOLO+SGBM 测距是纯本地推理，完全离线可用",
            '- LLM 请求失败会语音播报"网络失败"，不会卡住应用',
            "- 最核心的障碍物预警功能在断网情况下依然正常工作",
        ]
    ),
    (
        "跌倒会误判吗（比如跑步、跳跃）?",
        [
            "这是一个权衡设计。剧烈运动理论上可能触发，但有多个保护机制：",
            '- 跑步时失重期非常短，不容易同时满足"失重时间+1秒内撞击"条件',
            "- 手机被剧烈甩动后1.5秒无撞击，自动重置不误报",
            "- 误报成本低（播报一次警报），漏报成本高（用户跌倒无人知晓）",
        ]
    ),
    (
        "模型是你们自己训练的吗？",
        [
            "是的。基于 YOLOv8 框架自采了视障场景数据集（人行道、行人、常见障碍物），使用 Python + Ultralytics 框架训练，最后导出为 TFLite 格式并做 INT8 量化，使其能在手机 CPU 上实时运行。",
        ]
    ),
]

for q, ans_lines in challenge_qa:
    add_qa(doc, q, ans_lines)

# 结尾
for _ in range(3):
    doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
r = p.add_run("整理于 2026年5月 | 根据项目实际代码生成")
set_font(r, size=9, color=(136,136,136), italic=True)

output_path = r"d:\MyApplication1\答辩材料_声途智行.docx"
doc.save(output_path)
print(f"DOCX generated: {output_path}")
