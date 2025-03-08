import os
import matplotlib.pyplot as plt
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import detectron2.model_zoo
from detectron2.data.datasets import register_coco_instances

# -------------------------------
# Datasetの登録
# -------------------------------
register_coco_instances("manga_q_train", {}, "./manga109_train_coco.json", "./train_images")
register_coco_instances("manga_q_val", {}, "./manga109_val_coco.json", "./val_images")

# -------------------------------
# カスタムフック：各イテレーションごとに学習率を記録する
# -------------------------------
class LrLogger(HookBase):
    def __init__(self):
        self.lr_history = []
        self.iter_history = []
        
    def after_step(self):
        # オプティマイザの1つ目のパラメータグループの学習率を記録
        lr = self.trainer.optimizer.param_groups[0]['lr']
        self.lr_history.append(lr)
        self.iter_history.append(self.trainer.iter)

# -------------------------------
# カスタムフック：一定イテレーション毎（ここでは5000イテレーション毎）に
# mAPとトレーニング損失を記録する
# -------------------------------
class EvalAndLossLogger(HookBase):
    def __init__(self, eval_interval, cfg):
        """
        :param eval_interval: 何イテレーションごとに評価を行うか
        :param cfg: detectron2の設定オブジェクト
        """
        self.eval_interval = eval_interval
        self.cfg = cfg
        self.iterations = []
        self.map_history = []
        self.loss_history = []
    
    def after_step(self):
        # 現在のイテレーション+1が eval_interval の倍数なら評価
        if (self.trainer.iter + 1) % self.eval_interval == 0:
            current_iter = self.trainer.iter + 1
            print(f"Evaluating at iteration {current_iter} ...")
            
            # 検証用のEvaluatorとDataLoaderの作成
            evaluator = COCOEvaluator("manga_q_val", self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
            val_loader = build_detection_test_loader(self.cfg, "manga_q_val")
            eval_results = inference_on_dataset(self.trainer.model, val_loader, evaluator)
            
            # mAPの取得（"bbox" キー内の "AP" を想定）
            if "bbox" in eval_results and "AP" in eval_results["bbox"]:
                mAP = eval_results["bbox"]["AP"]
            elif "AP" in eval_results:
                mAP = eval_results["AP"]
            else:
                mAP = None  # 評価結果が想定と異なる場合
            
            # トレーニング損失の取得（型変換を試みる）
            loss_val = None  # 初期化しておく
            try:
                loss_val = self.trainer.storage.latest_with_smoothing_hint("total_loss")
                # 数値型でなければ float に変換
                if not isinstance(loss_val, (float, int)):
                    loss = float(loss_val)
                else:
                    loss = loss_val
            except Exception as e:
                print(f"Warning: Could not retrieve total_loss as a float: {loss_val} ({e})")
                loss = float('nan')  # NaNにすることで後でプロット時に無視される
            
            self.iterations.append(current_iter)
            self.map_history.append(mAP)
            self.loss_history.append(loss)
            
            print(f"Iteration {current_iter}: mAP = {mAP}, Training Loss = {loss}")
    
    def after_train(self):
        # 学習終了後にグラフを出力
        
        # --- mAPのグラフ ---
        plt.figure(figsize=(8, 6))
        plt.plot(self.iterations, self.map_history, marker="o", label="mAP")
        plt.xlabel("Iteration")
        plt.ylabel("mAP")
        plt.title("mAP over Iterations")
        plt.grid(True)
        plt.legend()
        mAP_fig_path = os.path.join(self.cfg.OUTPUT_DIR, "mAP_over_iterations.png")
        plt.savefig(mAP_fig_path)
        plt.show()
        print("mAPのグラフを保存しました:", mAP_fig_path)
        
        # --- トレーニング損失のグラフ ---
        plt.figure(figsize=(8, 6))
        plt.plot(self.iterations, self.loss_history, marker="o", color="red", label="Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Training Loss")
        plt.title("Training Loss over Iterations")
        plt.grid(True)
        plt.legend()
        loss_fig_path = os.path.join(self.cfg.OUTPUT_DIR, "TrainingLoss_over_iterations.png")
        plt.savefig(loss_fig_path)
        plt.show()
        print("トレーニング損失のグラフを保存しました:", loss_fig_path)

# -------------------------------
# 設定ファイルの読み込みと各種パラメータの設定
# -------------------------------
cfg = get_cfg()
cfg.merge_from_file(detectron2.model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("manga_q_train",)
cfg.DATASETS.TEST = ("manga_q_val",)
cfg.DATALOADER.NUM_WORKERS = 2
# モデルズーから初期重みをロード
cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4  # ミニバッチサイズ：画像4枚
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 106600  # 50エポック分（例：2132イテレーション x 50）
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # クラス数（顔検出なので1クラス）

# 出力ディレクトリの作成
cfg.OUTPUT_DIR = "./output_thesis"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# -------------------------------
# トレーナーの作成
# -------------------------------
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

# 学習率を記録するフックを登録
lr_logger = LrLogger()
trainer.register_hooks([lr_logger])

# mAP とトレーニング損失を 5000 イテレーション毎に記録するフックを登録
eval_loss_logger = EvalAndLossLogger(eval_interval=5000, cfg=cfg)
trainer.register_hooks([eval_loss_logger])

# -------------------------------
# 学習の実行
# -------------------------------
trainer.train()

# -------------------------------
# 学習終了後の処理：学習率スケジュールのグラフ作成
# -------------------------------
plt.figure(figsize=(8, 4))
plt.plot(lr_logger.iter_history, lr_logger.lr_history, label="Learning Rate")
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True)
lr_fig_path = os.path.join(cfg.OUTPUT_DIR, "lr_schedule.png")
plt.savefig(lr_fig_path)
plt.show()
print("学習率スケジュールのグラフを保存しました:", lr_fig_path)

# -------------------------------
# 最終評価
# -------------------------------
evaluator = COCOEvaluator("manga_q_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "manga_q_val")
final_eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)
print("Final Evaluation Results:")
print(final_eval_results)

# -------------------------------
# 学習済みモデルのファイルをアウトプット
# -------------------------------
final_model_path = os.path.join(cfg.OUTPUT_DIR, "final_model.pth")
torch.save(trainer.model.state_dict(), final_model_path)
print("Final model saved at:", final_model_path)
