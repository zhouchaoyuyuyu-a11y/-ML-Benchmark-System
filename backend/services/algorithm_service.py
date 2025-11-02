import json
import numpy as np
import logging
from algorithms.ml10 import serve_request
from dataset_loader import load_dataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)


def validate_algorithm_dataset_compatibility(algorithm, dataset_name, X, y):
    """
    验证算法与数据集的兼容性
    """
    # 分类算法需要分类数据集
    classification_algorithms = ['decision_tree_clf', 'svm_clf', 'knn_clf', 'gaussian_nb', 'logreg_clf', 'mlp_clf']
    regression_algorithms = ['linear_reg', 'mlp_reg']
    unsupervised_algorithms = ['kmeans', 'pca']

    classification_datasets = ['iris', 'wine', 'breast_cancer', 'digits', 'blobs', 'classification']
    regression_datasets = ['diabetes', 'california_housing', 'regression']

    # 无监督算法可以用于任何数据集（不需要标签）
    if algorithm in unsupervised_algorithms:
        return True, "验证通过"

    if algorithm in classification_algorithms and dataset_name not in classification_datasets:
        return False, f"分类算法 '{algorithm}' 不能用于回归数据集 '{dataset_name}'"

    if algorithm in regression_algorithms and dataset_name not in regression_datasets:
        return False, f"回归算法 '{algorithm}' 不能用于分类数据集 '{dataset_name}'"

    # 检查数据有效性
    if X is None or len(X) == 0:
        return False, "特征数据 X 为空"

    if y is None or len(y) == 0:
        return False, "标签数据 y 为空"

    return True, "验证通过"

def handle_algorithm_request(request_data: dict) -> dict:
    """
    对接前端请求与ml10算法模块的核心服务函数
    """
    try:
        # 1. 解析前端请求参数
        algo_name = request_data["algorithm"]
        action = request_data["action"]
        dataset_name = request_data["dataset"]
        params = request_data.get("params", {})
        state = request_data.get("state", None)

        logger.info(f"处理算法请求 - 算法: {algo_name}, 动作: {action}, 数据集: {dataset_name}")

        # 2. 加载数据集 - 对于所有动作都需要数据集
        X, y_true = load_dataset(dataset_name)

        # 3. 验证算法与数据集的兼容性
        is_valid, validation_msg = validate_algorithm_dataset_compatibility(algo_name, dataset_name, X, y_true)
        if not is_valid:
            return {"code": 400, "message": f"算法与数据集不兼容: {validation_msg}", "data": {}}

        # 对于推理和评估动作，需要模型状态
        if action in ["infer", "score"] and not state:
            return {"code": 400, "message": f"{action}动作需要模型状态，但未提供", "data": {}}

        # 转换数据格式
        X_list = X.tolist()
        y_true_list = y_true.tolist() if y_true is not None else None

        # 4. 构造ml10.py的serve_request输入
        ml10_payload = {
            "algo": algo_name,
            "action": action,
            "X": X_list,
            "y": y_true_list,
            "params": params,
            "state": state
        }

        # 5. 调用组员A的算法模块
        ml10_result = serve_request(ml10_payload)

        if not ml10_result["ok"]:
            return {"code": 400, "message": f"算法运行失败：{ml10_result['message']}", "data": {}}

        # 6. 整理基础结果
        response_data = {
            "code": 200,
            "message": "success",
            "data": {
                "basic_info": {
                    "algorithm": algo_name,
                    "dataset": dataset_name,
                    "task_type": ml10_result["task_type"],
                    "action": action
                },
                "metrics": ml10_result["metrics"] or {},
                "y_pred": ml10_result["y_pred"] or [],
                "y_proba": ml10_result["y_proba"] or [],
                "state": ml10_result["state"] or None,
                # 新增字段：用于增强可视化
                "y_true": y_true_list,  # 真实标签
                "dataset_info": {
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    "n_classes": len(np.unique(y_true)) if y_true is not None else None
                },
                # 初始化可视化所需的数据字段
                "confusion_matrix": None,
                "classification_report": None,
                "class_accuracy": None,
                "residuals": None,
                "prediction_pairs": None,
                "cluster_centers": None
            }
        }

        # 7. 针对特殊算法补充差异化结果
        if algo_name == "pca" and action == "train":
            # 从ml10的PCA模型state中提取解释方差比
            if ml10_result["state"] and "mean" in ml10_result["state"] and "var" in ml10_result["state"]:
                pca_mean = ml10_result["state"]["mean"]
                pca_var = ml10_result["state"]["var"]
                n_components = params.get("n_components", 2)
                explained_variance_ratio = [var / sum(pca_var) for var in pca_var[:n_components]]
                cumulative_var = np.cumsum(explained_variance_ratio).tolist()

                response_data["data"]["metrics"]["explained_variance_ratio"] = explained_variance_ratio
                response_data["data"]["metrics"]["cumulative_explained_variance"] = cumulative_var

        elif algo_name == "kmeans" and action == "train":
            # 提取KMeans聚类中心
            if ml10_result["state"] and "centers" in ml10_result["state"]:
                # 转换为浮点数列表，确保JSON序列化
                centers = ml10_result["state"]["centers"]
                if isinstance(centers, np.ndarray):
                    response_data["data"]["cluster_centers"] = centers.tolist()
                else:
                    response_data["data"]["cluster_centers"] = [[float(val) for val in center] for center in centers]
                    
                # 为聚类任务添加可视化所需的预测标签
                if ml10_result["y_pred"] is not None:
                    response_data["data"]["cluster_labels"] = ml10_result["y_pred"]

        # 8. 计算增强的可视化数据
        if action in ["train", "score"] and y_true is not None and ml10_result["y_pred"] is not None:
            y_pred_np = np.array(ml10_result["y_pred"])

            # 分类任务的可视化数据
            if ml10_result["task_type"] == "classification":
                try:
                    # 计算混淆矩阵
                    cm = confusion_matrix(y_true, y_pred_np)
                    response_data["data"]["confusion_matrix"] = cm.tolist()

                    # 计算分类报告 - 确保中文标签正常显示
                    class_report = classification_report(y_true, y_pred_np, output_dict=True)
                    # 转换数值类型以便JSON序列化
                    for key in class_report:
                        if isinstance(class_report[key], dict):
                            for sub_key in class_report[key]:
                                class_report[key][sub_key] = float(class_report[key][sub_key])
                    response_data["data"]["classification_report"] = class_report

                    # 计算各类别的准确率
                    unique_classes = np.unique(y_true)
                    class_accuracy = {}
                    for cls in unique_classes:
                        mask = y_true == cls
                        if np.sum(mask) > 0:
                            class_accuracy[int(cls)] = float(np.mean(y_pred_np[mask] == cls))
                    response_data["data"]["class_accuracy"] = class_accuracy
                    
                    # 增强分类指标
                    if "accuracy" not in response_data["data"]["metrics"]:
                        response_data["data"]["metrics"]["accuracy"] = float(accuracy_score(y_true, y_pred_np))
                    
                    # 添加精确率、召回率、F1分数
                    if len(unique_classes) > 2:  # 多分类
                        avg_method = "macro"
                        response_data["data"]["metrics"]["precision"] = float(precision_score(y_true, y_pred_np, average=avg_method))
                        response_data["data"]["metrics"]["recall"] = float(recall_score(y_true, y_pred_np, average=avg_method))
                        response_data["data"]["metrics"]["f1_score"] = float(f1_score(y_true, y_pred_np, average=avg_method))
                    else:  # 二分类
                        response_data["data"]["metrics"]["precision"] = float(precision_score(y_true, y_pred_np))
                        response_data["data"]["metrics"]["recall"] = float(recall_score(y_true, y_pred_np))
                        response_data["data"]["metrics"]["f1_score"] = float(f1_score(y_true, y_pred_np))
                    
                except Exception as e:
                    logger.error(f"分类任务可视化数据计算错误: {str(e)}")

            # 回归任务的可视化数据
            elif ml10_result["task_type"] == "regression":
                try:
                    # 计算残差
                    residuals = y_true - y_pred_np
                    response_data["data"]["residuals"] = residuals.tolist()

                    # 计算预测值与真实值的对应关系（用于散点图）
                    # 限制数据量，确保前端性能
                    max_pairs = min(100, len(y_true))
                    response_data["data"]["prediction_pairs"] = [
                        {"true": float(true_val), "pred": float(pred_val)}
                        for true_val, pred_val in zip(y_true[:max_pairs], y_pred_np[:max_pairs])
                    ]
                    
                    # 增强回归指标统计信息
                    response_data["data"]["metrics"]["residual_mean"] = float(np.mean(residuals))
                    response_data["data"]["metrics"]["residual_std"] = float(np.std(residuals))
                    response_data["data"]["metrics"]["residual_min"] = float(np.min(residuals))
                    response_data["data"]["metrics"]["residual_max"] = float(np.max(residuals))
                    
                except Exception as e:
                    logger.error(f"回归任务可视化数据计算错误: {str(e)}")

        logger.info(f"算法执行成功: {algo_name}, 动作: {action}, 指标: {response_data['data']['metrics']}")
        return response_data

    except Exception as e:
        logger.error(f"算法服务错误: {str(e)}")
        return {"code": 500, "message": f"服务端错误：{str(e)}", "data": {}}
