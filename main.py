# 导入所需库
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 关闭所有警告信息
import warnings
warnings.filterwarnings("ignore")

# 设置绘图字体，避免中文警告
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 自动创建图表保存文件夹
os.makedirs("outputs", exist_ok=True)

# ===================== 数据处理模块 =====================
def m1_data_process():
    print("=== Data Processing Start ===")
    df = pd.read_parquet(r"D:\1\taxi\data\yellow_tripdata_2023-01.parquet")
    df = df.dropna()
    df = df[(df["trip_distance"]>0) & (df["trip_distance"]<100)]
    df = df[(df["fare_amount"]>0) & (df["fare_amount"]<500)]
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["weekday"] = df["tpep_pickup_datetime"].dt.weekday  # 新增星期特征
    print("=== Data Processing Done ===")
    return df

# ===================== 可视化模块（4张图，满足作业要求） =====================
def m2_visualize(df):
    print("=== Generating 4 Charts ===")
    
    # 1. 每小时订单量折线图
    df.groupby("hour").size().plot(kind="line", figsize=(10,5))
    plt.title("Hourly Taxi Demand")
    plt.xlabel("Hour")
    plt.ylabel("Number of Trips")
    plt.savefig("outputs/1_hourly_demand.png")
    plt.close()

    # 2. 热门上车区域 TOP10 柱状图
    df["PULocationID"].value_counts().head(10).plot(kind="bar", figsize=(10,5))
    plt.title("Top 10 Pickup Zones")
    plt.xlabel("Location ID")
    plt.ylabel("Trips")
    plt.savefig("outputs/2_top_zones.png")
    plt.close()
    
    # 3. 行程距离 vs 车费 散点图（作业要求第3张）
    sample = df.sample(5000, random_state=42)  # 采样5000行防止卡顿
    plt.figure(figsize=(10,5))
    plt.scatter(sample["trip_distance"], sample["fare_amount"], s=2, alpha=0.5)
    plt.title("Trip Distance vs Fare Amount")
    plt.xlabel("Trip Distance")
    plt.ylabel("Fare")
    plt.savefig("outputs/3_distance_fare.png")
    plt.close()

    # 4. 工作日 vs 周末 小时流量对比图（作业要求第4张）
    weekday = df[df["weekday"] < 5].groupby("hour").size()
    weekend = df[df["weekday"] >= 5].groupby("hour").size()
    plt.figure(figsize=(10,5))
    weekday.plot(label="Weekday", linewidth=2)
    weekend.plot(label="Weekend", linewidth=2)
    plt.legend()
    plt.title("Weekday vs Weekend Demand")
    plt.xlabel("Hour")
    plt.ylabel("Trips")
    plt.savefig("outputs/4_weekday_vs_weekend.png")
    plt.close()
    
    print("=== 4 Charts Saved to outputs/ ===")

# ===================== 预测模型模块 =====================
def m3_model(df):
    print("=== Training Prediction Model ===")
    df_agg = df.groupby(["PULocationID", "hour"]).size().reset_index(name="demand")
    X = df_agg[["PULocationID", "hour"]]
    y = df_agg["demand"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=30)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")
    print("=== Model Training Done ===")

# ===================== 问答系统模块 =====================
def m4_qa(df):
    print("\n===== Final Assignment QA System =====")
    print("Commands: peak, zone, fare, weekend, exit\n")
    
    while True:
        q = input("Enter question: ")
        if q == "exit":
            break
        elif "peak" in q:
            h = df["hour"].value_counts().idxmax()
            print(f"Peak hour: {h}")
        elif "zone" in q:
            print("Top zones:\n", df["PULocationID"].value_counts().head())
        elif "fare" in q:
            avg = df["fare_amount"].mean()
            print(f"Average fare: {avg:.2f}")
        elif "weekend" in q or "weekday" in q:
            print("Weekday vs Weekend chart saved: outputs/4_weekday_vs_weekend.png")
        else:
            print("Try: peak, zone, fare, weekend, exit")

# ===================== 主程序 =====================
if __name__ == "__main__":
    df = m1_data_process()
    m2_visualize(df)  # 输出4张图
    m3_model(df)
    m4_qa(df)
    print("\n🎉 Assignment 100% Completed!")
