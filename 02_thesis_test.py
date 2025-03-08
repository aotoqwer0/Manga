import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- パスの設定 ---
# 検証用アノテーション（Ground Truth）の JSON ファイル
gt_json = "./manga109_val_coco.json"
# 出力ディレクトリ（元の学習コードの出力先）
output_dir = "./output_thesis"
# COCOEvaluator が出力した検出結果の JSON ファイル（通常はこのファイル名になります）
dt_json = os.path.join(output_dir, "inference/coco_instances_results.json")

# --- COCO API を用いた評価 ---
# 1. Ground Truth の読み込み
coco_gt = COCO(gt_json)
# 2. 検出結果の読み込み
coco_dt = coco_gt.loadRes(dt_json)

# 3. COCOeval オブジェクトの生成（bbox評価）
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

# 4. IoU の閾値を 0.50 のみとする（numpy 配列に変換）
coco_eval.params.iouThrs = np.array([0.5])

# 5. 評価の実行
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# --- 評価結果から IoU50 指標を取得 ---
# COCOeval.stats の内訳（COCOeval の仕様）:
# stats[1] : AP at IoU=0.50 (Precision)
# stats[8] : AR at IoU=0.50, maxDets=100 (Recall)
ap50 = coco_eval.stats[1]
ar50 = coco_eval.stats[8]

# F1 スコアの計算（調和平均）
f1 = 2 * ap50 * ar50 / (ap50 + ar50) if (ap50 + ar50) > 0 else 0

# --- 結果の表示 ---
print("\n==== IoU=0.50 における評価指標 ====")
print(f"Precision (AP50): {ap50:.4f}")
print(f"Recall    (AR50): {ar50:.4f}")
print(f"F1 Score:          {f1:.4f}")
