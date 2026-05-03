import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 导入数据集划分函数，用于将数据分为训练集和测试集
from sklearn.model_selection import train_test_split
# 导入随机森林回归模型，用于构建订单量预测模型
from sklearn.ensemble import RandomForestRegressor
# 导入模型评估指标，计算预测结果的误差
from sklearn.metrics import mean_absolute_error, mean_squared_error
# 导入警告管理库，关闭程序运行时的不必要提示
import warnings
warnings.filterwarnings("ignore")
# 设置绘图使用的默认英文字体，避免中文乱码问题
plt.rcParams["font.family"] = ["DejaVu Sans"]
# 设置图表负号正常显示，防止负号变为方框
plt.rcParams["axes.unicode_minus"] = False
# 自动创建保存图表的文件夹，如果已存在则不报错
os.makedirs("outputs", exist_ok=True)
# 数据处理函数 
def m1_data_process():
    """
    数据处理主函数，负责加载、清洗、提取时间特征
    返回清洗完成后的数据集，供后续模块使用
    """
    # 输出数据处理开始提示信息
    print("Data Processing Start")

    # 从指定绝对路径读取出租车数据集
    df = pd.read_parquet(r"D:\1\taxi\data\yellow_tripdata_2023-01.parquet")

    # 删除数据中存在缺失值的行，保证数据完整性
    df = df.dropna()

    # 过滤行程距离异常的数据，只保留距离在 0 到 100 之间的记录
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]

    # 过滤车费异常的数据，只保留车费在 0 到 500 之间的记录
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 500)]

    # 将上车时间字段转换为标准时间格式，方便提取时间特征
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    # 从上车时间中提取小时（0-23），用于分析时段订单量
    df["hour"] = df["tpep_pickup_datetime"].dt.hour

    # 从上车时间中提取星期几（0=周一，6=周日）
    df["weekday"] = df["tpep_pickup_datetime"].dt.weekday

    # 输出数据处理完成提示
    print("Data Processing Done")

    # 返回处理完成的数据集
    return df

#  可视化绘图函数 
def m2_visualize(df):
    """
    根据清洗后的数据生成四张分析图表并保存
    1. 每小时订单量折线图
    2. 热门上车区域柱状图
    3. 行程距离与车费关系散点图
    4. 工作日与周末订单对比图
    """
    # 输出图表生成提示
    print("Generating 4 Charts")
    
    # 第一张图：每小时订单量折线图
    df.groupby("hour").size().plot(kind="line", figsize=(10,5))
    plt.title("Hourly Taxi Demand")
    plt.xlabel("Hour")
    plt.ylabel("Number of Trips")
    plt.savefig("outputs/1_hourly_demand.png")
    plt.close()

    # 第二张图：热门上车区域TOP10柱状图
    df["PULocationID"].value_counts().head(10).plot(kind="bar", figsize=(10,5))
    plt.title("Top 10 Pickup Zones")
    plt.xlabel("Location ID")
    plt.ylabel("Trips")
    plt.savefig("outputs/2_top_zones.png")
    plt.close()
    
    # 第三张图：行程距离与车费散点图（采样5000条数据避免卡顿）
    sample = df.sample(5000, random_state=42)
    plt.figure(figsize=(10,5))
    plt.scatter(sample["trip_distance"], sample["fare_amount"], s=2, alpha=0.5)
    plt.title("Trip Distance vs Fare Amount")
    plt.xlabel("Trip Distance")
    plt.ylabel("Fare")
    plt.savefig("outputs/3_distance_fare.png")
    plt.close()

    # 第四张图：工作日与周末订单量对比图
    weekday_data = df[df["weekday"] < 5].groupby("hour").size()
    weekend_data = df[df["weekday"] >= 5].groupby("hour").size()
    plt.figure(figsize=(10,5))
    weekday_data.plot(label="Weekday", linewidth=2)
    weekend_data.plot(label="Weekend", linewidth=2)
    plt.legend()
    plt.title("Weekday vs Weekend Demand")
    plt.xlabel("Hour")
    plt.ylabel("Trips")
    plt.savefig("outputs/4_weekday_vs_weekend.png")
    plt.close()
    
    # 输出图表保存完成提示
    print("4 Charts Saved to outputs/")

#  预测模型训练函数 
def m3_model(df):
    """
    构建并训练随机森林预测模型
    根据区域ID和小时预测订单需求量
    输出模型评估指标 MAE 和 RMSE
    """
    print("Training Prediction Model")

    # 按区域ID和小时分组，统计每个组合的订单数量
    df_agg = df.groupby(["PULocationID", "hour"]).size().reset_index(name="demand")

    # 构建特征：区域ID和小时
    X = df_agg[["PULocationID", "hour"]]

    # 构建目标变量：订单量
    y = df_agg["demand"]
    
    # 将数据按8:2划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 创建随机森林回归模型
    model = RandomForestRegressor(n_estimators=30)

    # 使用训练集训练模型
    model.fit(X_train, y_train)
    
    # 使用测试集进行预测
    pred = model.predict(X_test)

    # 输出平均绝对误差
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")

    # 输出均方根误差
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")

    # 输出模型训练完成提示
    print("Model Training Done")

#  交互式问答系统
def m4_qa(df):
    """
    命令行交互式问答系统
    支持查询：高峰时段、热门区域、平均车费、周末对比、退出程序
    """
    print("\nFinal Assignment QA System")
    print("Commands: peak, zone, fare, weekend, exit\n")
    
    # 循环接收用户输入，实现持续交互
    while True:
        q = input("Enter question: ")

        # 输入 exit 退出程序
        if q == "exit":
            break

        # 查询高峰时段
        elif "peak" in q:
            h = df["hour"].value_counts().idxmax()
            print(f"Peak hour: {h}")

        # 查询热门区域
        elif "zone" in q:
            print("Top zones:\n", df["PULocationID"].value_counts().head())

        # 查询平均车费
        elif "fare" in q:
            avg = df["fare_amount"].mean()
            print(f"Average fare: {avg:.2f}")

        # 查询周末与工作日对比
        elif "weekend" in q or "weekday" in q:
            print("Weekday vs Weekend chart saved: outputs/4_weekday_vs_weekend.png")

        # 输入不支持的指令时给出提示
        else:
            print("Try: peak, zone, fare, weekend, exit")

#  程序主入口 
if __name__ == "__main__":
    # 执行数据处理
    df = m1_data_process()

    # 生成四张图表
    m2_visualize(df)

    # 训练预测模型
    m3_model(df)

    # 启动问答系统
    m4_qa(df)

    # 输出作业完成提示
    print("\nAssignment 100% Completed!")
