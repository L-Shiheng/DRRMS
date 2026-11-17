
# main.py
# DRRMS-0.1: Dynamic Relation-driven Model Superposition
# Supports tabular classification and regression.
# pip install streamlit pandas numpy scikit-learn

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

st.set_page_config(page_title="DRRMS-0.1", layout="wide")
st.title("DRRMS-0.1 动态模型叠加算法")
st.caption("与传统机器学习模型进行同数据集对照实验")

uploaded = st.file_uploader("上传 CSV 数据集", type=["csv"])

if uploaded is None:
    st.info("请上传 CSV 文件开始实验。")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("数据预览")
st.dataframe(df.head(10), use_container_width=True)

target = st.selectbox("选择预测目标列", df.columns)
task = st.radio("任务类型", ["自动判断", "分类", "回归"], horizontal=True)

test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
random_state = st.number_input("随机种子", 1, 9999, 42)
relation_strength = st.slider(
    "关系冗余惩罚系数", 0.0, 2.0, 0.3, 0.05
)
temperature = st.slider(
    "动态权重温度", 0.05, 3.0, 0.5, 0.05
)

def build_models(is_classification):
    if is_classification:
        return {
            "LogisticRegression": make_pipeline(
                SimpleImputer(strategy="median"),
                LogisticRegression(max_iter=1500)
            ),
            "RandomForest": make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=250, min_samples_leaf=2,
                    random_state=random_state, n_jobs=-1
                )
            ),
            "ExtraTrees": make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesClassifier(
                    n_estimators=250, min_samples_leaf=2,
                    random_state=random_state, n_jobs=-1
                )
            ),
            "HistGradientBoosting": make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(random_state=random_state)
            )
        }
    return {
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            Ridge(alpha=1.0)
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=250, min_samples_leaf=2,
                random_state=random_state, n_jobs=-1
            )
        ),
        "ExtraTrees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=250, min_samples_leaf=2,
                random_state=random_state, n_jobs=-1
            )
        ),
        "HistGradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(random_state=random_state)
        )
    }

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(np.clip(x, -50, 50))
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def dynamic_relation_weights(loss_hat, relation, strength, temp):
    """
    loss_hat: 每个样本、每个专家的预测损失
    relation: 专家间误差关联矩阵
    通过动态优化得到样本级模型权重。
    """
    n, m = loss_hat.shape
    weights = softmax(-loss_hat / temp)

    # 迭代优化：降低预测损失，同时抑制冗余模型
    step = 0.15
    for _ in range(25):
        grad = loss_hat + strength * (weights @ relation)
        weights = softmax(np.log(weights + 1e-12) - step * grad / temp)

    return weights

def run_experiment(X_train, X_test, y_train, y_test, classification):
    models = build_models(classification)
    names = list(models.keys())
    m = len(names)

    if classification:
        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=random_state
        )
    else:
        cv = KFold(
            n_splits=5, shuffle=True, random_state=random_state
        )

    oof_losses = []
    test_predictions = []

    # 第一阶段：交叉验证，估计训练样本上的专家误差
    for name in names:
        model = models[name]

        if classification:
            oof_prob = cross_val_predict(
                clone(model), X_train, y_train,
                cv=cv, method="predict_proba", n_jobs=-1
            )
            class_index = {
                c: i for i, c in enumerate(np.unique(y_train))
            }
            true_index = np.array([class_index[v] for v in y_train])
            sample_loss = -np.log(
                np.clip(oof_prob[np.arange(len(y_train)), true_index],
                        1e-8, 1.0)
            )
        else:
            oof_pred = cross_val_predict(
                clone(model), X_train, y_train,
                cv=cv, method="predict", n_jobs=-1
            )
            sample_loss = (np.asarray(y_train) - oof_pred) ** 2

        oof_losses.append(sample_loss)

        fitted = clone(model)
        fitted.fit(X_train, y_train)

        if classification:
            test_predictions.append(fitted.predict_proba(X_test))
        else:
            test_predictions.append(fitted.predict(X_test))

    # n_samples × n_experts
    loss_matrix = np.column_stack(oof_losses)

    # 误差标准化，避免不同专家误差尺度影响关系估计
    loss_mean = loss_matrix.mean(axis=0)
    loss_std = loss_matrix.std(axis=0) + 1e-8
    normalized_loss = (loss_matrix - loss_mean) / loss_std

    # 第二阶段：学习不同输入条件下各专家的预期误差
    gate_models = []
    predicted_losses = []

    for j in range(m):
        gate = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=120,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=random_state + j,
                n_jobs=-1
            )
        )
        gate.fit(X_train, normalized_loss[:, j])
        gate_models.append(gate)
        predicted_losses.append(gate.predict(X_test))

    predicted_losses = np.column_stack(predicted_losses)

    # 第三阶段：建立专家之间的误差关系
    relation = np.corrcoef(normalized_loss, rowvar=False)
    relation = np.nan_to_num(relation, nan=0.0)
    relation = np.maximum(relation, 0.0)
    np.fill_diagonal(relation, 0.0)

    # 第四阶段：动态关系叠加
    weights = dynamic_relation_weights(
        predicted_losses, relation,
        relation_strength, temperature
    )

    # 输出 DRRMS 预测
    if classification:
        stacked = np.stack(test_predictions, axis=1)
        drrms_prob = np.sum(stacked * weights[:, :, None], axis=1)
        drrms_pred = np.argmax(drrms_prob, axis=1)
    else:
        stacked = np.column_stack(test_predictions)
        drrms_pred = np.sum(stacked * weights, axis=1)

    # 传统模型与 DRRMS 的统一评价
    results = []

    for j, name in enumerate(names):
        pred = (
            np.argmax(test_predictions[j], axis=1)
            if classification else test_predictions[j]
        )
        results.append((name, pred))

    results.append(("DRRMS-0.1", drrms_pred))

    rows = []
    for name, pred in results:
        if classification:
            rows.append({
                "模型": name,
                "Accuracy": accuracy_score(y_test, pred),
                "F1_macro": f1_score(
                    y_test, pred, average="macro", zero_division=0
                )
            })
        else:
            rows.append({
                "模型": name,
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "MAE": mean_absolute_error(y_test, pred),
                "R2": r2_score(y_test, pred)
            })

    return pd.DataFrame(rows), weights, relation, names

if st.button("开始训练与对比", type="primary"):
    try:
        data = df.dropna(subset=[target]).copy()
        X = data.drop(columns=[target])
        y = data[target]

        # 暂时只处理数值特征，避免把任意类别编码误当作连续数值
        X = pd.get_dummies(X, dummy_na=True)
        X = X.replace([np.inf, -np.inf], np.nan)

        if task == "分类":
            classification = True
        elif task == "回归":
            classification = False
        else:
            classification = (
                y.dtype == "object"
                or y.nunique() <= 20
            )

        if classification:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y.astype(str))
            if len(np.unique(y)) < 2:
                st.error("分类目标至少需要两个类别。")
                st.stop()
            if pd.Series(y).value_counts().min() < 5:
                st.error("最少类别样本数不足5，无法进行5折交叉验证。")
                st.stop()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size,
                random_state=int(random_state), stratify=y
            )
        else:
            y = pd.to_numeric(y, errors="coerce")
            valid = y.notna()
            X, y = X.loc[valid], y.loc[valid]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size,
                random_state=int(random_state)
            )

        with st.spinner("正在进行交叉验证、关系学习和模型对比..."):
            result, weights, relation, names = run_experiment(
                X_train, X_test, y_train, y_test, classification
            )

        st.success("实验完成")

        st.subheader("一、模型性能对比")
        st.dataframe(
            result.sort_values(
                "F1_macro" if classification else "RMSE",
                ascending=not classification
            ),
            use_container_width=True
        )

        st.subheader("二、动态模型权重")
        weight_df = pd.DataFrame(weights, columns=names)
        st.dataframe(weight_df.describe(), use_container_width=True)

        st.subheader("三、模型关系矩阵")
        st.caption("数值越高，代表两个模型的误差越倾向于同步变化。")
        st.dataframe(
            pd.DataFrame(relation, index=names, columns=names),
            use_container_width=True
        )

        st.subheader("四、实验参数")
        st.write({
            "任务类型": "分类" if classification else "回归",
            "训练样本数": len(X_train),
            "测试样本数": len(X_test),
            "特征数量": X.shape[1],
            "关系惩罚系数": relation_strength,
            "动态权重温度": temperature
        })

        st.download_button(
            "下载性能对比结果 CSV",
            result.to_csv(index=False).encode("utf-8-sig"),
            "DRRMS_comparison.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"实验失败：{e}")
        st.exception(e)
