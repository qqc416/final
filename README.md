人机协作报告（纽约出租车出行数据分析与智能问答系统）

包含：分步拆解Prompt、交互日志、AI犯错Debug案例、三阶段代码对比、反思

 我完成任务的步骤

1. 新建项目结构：data、outputs、main.py

2. 下载纽约出租车2023-01.parquet数据

3. 安装依赖：pandas、matplotlib、scikit-learn等

4. 分步向AI提问，逐个完成M1–M4

5. 运行代码，遇到错误后再次向AI提问修复

6. 完成图表保存、模型训练、问答系统

7. 使用ai添加注释，并仔细阅读

一、任务整体拆解

我把作业拆成4个模块，逐个向AI提问：

1. M1 数据处理：加载、清洗、特征工程

2. M2 可视化：4张图表绘制与保存

3. M3 预测模型：随机森林+需求预测

4. M4 问答系统：命令行关键词匹配

 二、AI 交互日志

每模块均以单独Prompt提问。

 模块1：M1 数据处理

 我的任务拆解

- 读取parquet数据

- 输出缺失率、异常值报告

- 清洗：去空值、去异常距离/车费

- 提取：hour、weekday、is_peak

 我的Prompt

帮我写M1数据处理代码，加载数据，生成数据质量报告（缺失率、异常值统计）；清洗数据并在注释中说明每步策略的理由；从行程时间中提取小时、星期、是否高峰等特征，并自行设计至少 2 个有意义的衍生特征。：

1. 用pandas读取yellow_tripdata_2023-01.parquet

2. 输出每列缺失率百分比

3. 删除缺失行，过滤trip_distance 0~100、fare_amount 0~500

4. 把tpep_pickup_datetime转时间，提取hour、weekday

5. 添加两个衍生特征：车速、每公里车费

6. 代码简洁可运行，不要多余内容

 AI真实错误与Debug（必须写）

错误类型：逻辑顺序错误（作业要求先报告后清洗）

错误现象：先dropna()再计算缺失率，结果全0，无法生成质量报告

发现方式：运行后控制台无真实缺失率输出

修正方法：将问题投喂给ai，让其进行修改

模块2：M2 分析可视化

 任务拆解

- 小时需求折线图

- 热门区域TOP10柱状图

- 距离-车费散点图

- 周末vs工作日对比图

- 保存到outputs

我的Prompt


帮我写M2可视化，只画4张图：

1. 按小时统计订单量折线图

2. PULocationID TOP10柱状图

3. trip_distance与fare_amount散点图（采样5000行）

4. 工作日/周末每小时对比图

全部保存到outputs给出4张图代码，plt保存。

 AI真实错误与Debug

错误类型：matplotlib中文字体警告

错误现象：运行出现大量 `Glyph missing` 红字

发现方式：控制台刷屏

修正方法：关闭中文、设置英文字体、关闭警告

修正Promp;把图表全部改成英文标题，关闭matplotlib中文，消除字体警告


模块3：M3 预测模型

 任务拆解

- 按区域+小时聚合需求量

- 8:2划分训练/测试集

- 随机森林训练

- 输出MAE、RMSE

我的Prompt

帮我写M3预测模型，只做需求预测：

1. 按PULocationID+hour分组统计订单数demand

2. 特征用LocationID和hour

3. 用随机森林回归，8:2分割

4. 打印MAE、RMSE

不要写PyTorch，只保留sklearn

AI真实错误与Debug

错误类型：库缺失错误

错误现象：ModuleNotFoundError: sklearn不存在

发现方式：运行直接报错

修正方法：安装scikit-learn

修正Prompt：给出本代码需要安装的所有依赖库名称


模块4：M4 智能问答系统
 任务拆解

- 命令行循环

- 支持5类问题：高峰、区域、费用、预测、周末

- 关键词匹配 + 返回结果

 我的Prompt

帮我写M4问答系统，命令行交互：

1. 支持5类问题：高峰小时、热门区域、平均车费、需求预测、周末对比

2. 用if-else关键词匹配

3. 输入exit退出

4. 输出结果+图表路径

AI真实错误与Debug

错误类型：逻辑冲突（多条件同时触发）

错误现象：输入“高峰区域”同时输出两个答案

发现方式：测试交互时出现

修正方法：使用ai让其避免逻辑冲突

三、三阶段代码对比（以 M1 数据处理 为例）
 1. Native 
import pandas as pd
df = pd.read_parquet("data/yellow_tripdata_2023-01.parquet")
print(df.isnull().sum() / len(df) * 100)
df = df.dropna()
df = df[(df["trip_distance"]>0) & (df["trip_distance"]<100)]
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["hour"] = df["tpep_pickup_datetime"].dt.hour

- 效率：慢，需查文档

- 理解深度：极高，完全掌握每一步

 2. Prompt 
import pandas as pd
def m1():
    df = pd.read_parquet("data/...parquet")
    miss = df.isnull().sum()/len(df)*100
    print(miss)
    df = df.dropna()
    df["hour"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.hour
    return df

- 效率：快

- 理解深度：中等，能看懂但细节依赖AI
  
3. Vibe 

def m1_data_process():
    df = pd.read_parquet(r"D:\1\taxi\data\...parquet")
    print("缺失率:\n", df.isnull().sum()/len(df)*100)
    df = df.dropna()
    df = df[(df["trip_distance"]>0) & (df["trip_distance"]<100)]
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["speed"] = ... 
    return df

- 效率：最高

- 理解深度：高，能定位错误并优化



#四、反思
通过本次作业，我深刻认识到AI是高效辅助工具而非替代者。AI能快速生成代码框架、处理重复逻辑、解释报错信息，但它无法真正理解作业评分点、业务约束与运行环境。
AI多次出现逻辑顺序错误、库缺失、字体冲突、条件重叠等问题，必须由人来判断、修正、调试。AI擅长“实现”，不擅长“审题、落地、排错”。它不能替代思考，只能提升效率。未来我会继续用AI辅助编码
，但会坚持先拆解任务、分步提问、严格校验输出，确保代码真实、可运行、符合要求。





