import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import slideflow as sf
    import matplotlib.pyplot as plt
    import pandas as pd
    from typing import Optional, Dict, Any, Tuple, Union, List
    from transformers import get_cosine_schedule_with_warmup

    # PyTorch / Lightning imports
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchmetrics.functional import accuracy, precision, recall, f1_score, auroc
    from torch.optim import AdamW
    import pytorch_lightning as pl

    from utils.sf2torch import DatasetFromSlideFlow
    from utils.mil_lightning import MILDataModule
    return (
        AdamW,
        Any,
        DatasetFromSlideFlow,
        Dict,
        F,
        List,
        MILDataModule,
        Optional,
        Tuple,
        Union,
        accuracy,
        auroc,
        f1_score,
        get_cosine_schedule_with_warmup,
        mo,
        nn,
        pl,
        precision,
        recall,
        sf,
        torch,
    )


@app.cell
def _(DatasetFromSlideFlow, MILDataModule, sf):
    P = sf.load_project("/workspace/slideflow_project")
    dataset = P.dataset(tile_px=224, tile_um='10x', filters={"category": ["tumor", "normal"]})
    # dataset = P.dataset(tile_px=224, tile_um='10x', filters={"sample": "True"})

    train_sf_dataset, valid_sf_dataset = dataset.split(
        model_type='classification', # Categorical labels
        labels='category',            # Label to balance between datasets
        val_strategy='fixed',          # Fixed split strategy
        val_fraction=0.2,             # 20% of data for testing
        splits='/workspace/slideflow_project/splits.json'         # Where to save/load crossfold splits
    )

    test_sf_dataset = P.dataset(tile_px=224, tile_um='10x', filters={"category": ["test"]})

    train_torch_dataset = DatasetFromSlideFlow(train_sf_dataset)
    valid_torch_dataset = DatasetFromSlideFlow(valid_sf_dataset)
    test_torch_dataset = DatasetFromSlideFlow(test_sf_dataset)

    data_module = MILDataModule(train_torch_dataset, valid_torch_dataset, test_torch_dataset, batch_size=4, num_workers=2)
    return (
        data_module,
        test_torch_dataset,
        train_torch_dataset,
        valid_torch_dataset,
    )


@app.cell
def _(F, nn, torch):
    class AttentionMIL(nn.Module):
        def __init__(self, embed_dim, attn_dim, output_dim, dropout_rate=0.5):
            super().__init__()
        
            # V: tanch branch
            self.V = nn.Sequential(
                nn.Linear(embed_dim, attn_dim),
                nn.Tanh()
            )

            # H: sigmoid branch
            self.U = nn.Sequential(
                nn.Linear(embed_dim, attn_dim),
                nn.Sigmoid()
            )   
        
            # W: Attention係数 (A) の線形変換
            self.W = nn.Sequential(
                nn.Linear(attn_dim, 1),
            )

            self.dropout = nn.Dropout(p=dropout_rate)

            # バッグレベルの分類器
            self.classifier = nn.Linear(embed_dim, output_dim) #logitsで出力

        def forward(self, X_total, N_sizes):
            # X_total: (N_total, D) - 連結された全インスタンス特徴
            # N_sizes: (B,) - 各バッグのインスタンス数

            V = self.V(X_total)  # (N_total, attn_dim)
            U = self.U(X_total)  # (N_total, attn_dim)
            VU = V * U            # (N_total, attn_dim)

            # softmax前のAttentionスコア
            A_raw = self.W(VU)     # (N_total, 1)

            # バッグごとの集約とAttentionの適用 (重要部分)
        
            # X_totalとA_rawをバッグごとに分割するためのオフセットを作成
            # cumsum(N_sizes) は各バッグの終了インデックスを示す
        
            all_bag_representations = []
            all_bag_attentions = []
            start_idx = 0

            for i, N_i in enumerate(N_sizes):
                end_idx = start_idx + N_i.item()
            
                H_i = X_total[start_idx:end_idx]  # (N_i, embed_dim) - i番目のバッグのインスタンス特徴
                A_i_raw = A_raw[start_idx:end_idx] # (N_i, 1) - i番目のバッグのAttentionスコア
            
                # Attentionスコアのsoftmax正規化
                A_i = F.softmax(A_i_raw, dim=0) # Attention weights: (N_i, 1)

                # Attention Weighted Bag Representation: (1, embed_dim)
                # Z = A^T * H_i
                Z_i = torch.transpose(A_i, 0, 1) @ H_i # (1, N_i) @ (N_i, embed_dim) -> (1, embed_dim)
            
                all_bag_representations.append(Z_i.squeeze(0))
                all_bag_attentions.append(A_i.squeeze(-1)) # (N_i,)
            
                start_idx = end_idx # 次のバッグの開始インデックスを更新

            Z_batch = torch.stack(all_bag_representations, dim=0)  # すべてのバッグ表現をスタック: (B, attn_dim)
            A_batch = all_bag_attentions  # リスト of (N_i,) tensors
        
            Z_batch = self.dropout(Z_batch)

            # 4. バッグレベルの分類
            Y_logits = self.classifier(Z_batch) # (B, output_dim)

            return {"logits": Y_logits, "bag_representation": Z_batch, "attention": A_batch}
    return (AttentionMIL,)


@app.cell
def _(AttentionMIL, data_module):
    dl = data_module.train_dataloader()
    X_total, N_sizes, Y_labels = next(iter(dl))

    # モデルのインスタンス化
    embed_dim = 2048
    attn_dim = 256
    output_dim = 1
    model = AttentionMIL(embed_dim, attn_dim, output_dim, dropout_rate=0.5)

    # モデルの使用例 (forward pass)
    Y_logits, Z_batch, A_batch = model(X_total, N_sizes).values()

    print(f"\nモデルの出力の形状 (Y_logits): {Y_logits.shape}")
    print(f"集約されたバッグ表現の形状 (Z_batch): {Z_batch.shape}")
    return


@app.cell
def _(
    AdamW,
    Any,
    AttentionMIL,
    Dict,
    F,
    List,
    Optional,
    Tuple,
    Union,
    accuracy,
    auroc,
    f1_score,
    get_cosine_schedule_with_warmup,
    pl,
    precision,
    recall,
    torch,
):
    class AttentionMILLitModule(pl.LightningModule):
        """Attention MIL の LightningModule"""

        def __init__(
            self,
            num_classes: int = 2,
            embed_dim: int = 2048,
            attn_dim: int = 128,
            dropout: float = 0.5,
            lr: float = 1e-4,
            pos_weight: Optional[Union[float, List[float]]] = None,  # class imbalance 用
            eps=1e-8, # AdamWでよく使われる小さなepsilon
            warmup_steps: int = 500,
            max_steps: int = 5000,
        ):
            super().__init__()
            self.save_hyperparameters()

            self.attention_mil = AttentionMIL(
                embed_dim=embed_dim,
                attn_dim=attn_dim,
                output_dim=num_classes if num_classes > 2 else 1,
                dropout_rate=dropout,
            )

            self.num_classes = num_classes

            if num_classes > 2:
                # multi-class
                if pos_weight is not None:
                    assert len(pos_weight) == num_classes, "pos_weight length must match num_classes"
                    self.pos_weight = torch.Tensor(pos_weight)
            else:
                # binary
                if pos_weight is not None:
                    assert isinstance(pos_weight, (float, int)), "pos_weight must be a single float for binary classification"
                    self.pos_weight = torch.Tensor([pos_weight])

        def forward(self, X_total: torch.Tensor, N_sizes: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            推論用: X_total: [N_total, in_dim]
            Returns:
                {
                  "logits": [B, num_classes or 1],
                  "probs":  [B, num_classes or 1],
                  "attention":   [List of (N_i,) tensors],
                }
            """
            out = self.attention_mil(X_total=X_total, N_sizes=N_sizes)  # bag_rep, attn, H
            logits = out["logits"]
            bag_reprsentations = out["bag_repr"]
            attention = out["attention"]

            if self.num_classes == 1:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)

            return {
                "logits": logits,
                "probs": probs,
                "attention": attention,
                "bag_reprsentations": bag_reprsentations,
            }

        # ----------------------------
        # Loss & metrics
        # ----------------------------
        def _compute_loss(
            self, logits: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            """
            y:
              - binary: [B] or [B, 1] (0/1)
              - multi:  [B] (int)
            """

            if self.num_classes > 2:
                # multi-class
                assert y.ndim == 1, "For multi-class classification, y should be of shape [B] (one-hot not supported)"
                loss = F.cross_entropy(
                    input=logits, 
                    target=y.long(), 
                    weight=self.pos_weight.to(self.device) if self.pos_weight is not None else None
                )
            else:
                # binary
                # inputとtargetのshapeを合わせる必要がある→yの次元が1ならunsqueeze
                if y.ndim == 1:
                    y = y.unsqueeze(-1)
                assert y.ndim == 2, "For binary classification, y should be of shape [B, 1]"
                loss = F.binary_cross_entropy_with_logits(
                    input=logits, 
                    target=y.float(), 
                    pos_weight=self.pos_weight.to(self.device) if self.pos_weight is not None else None
                )
            return loss

        def _compute_metrics(
            self, logits: torch.Tensor, y: torch.Tensor
        ) -> dict:
            """
            精度 (accuracy)、適合率 (precision)、再現率 (recall)、F1スコア (f1-score) を計算します。
            torchmetrics を使用します
        
            Args:
                logits (torch.Tensor): モデルの出力（ロジット）。
                y (torch.Tensor): 真のラベル。

            Returns:
                dict: 各評価指標を含む辞書。
            """
        
            # 結果を格納するための辞書
            metrics = {}
        
            # 1. 予測値の準備
            if self.num_classes > 2:
                # multi-class
                task = "multiclass"
                preds_for_metrics = logits # multi-class→ロジットを渡す
                assert y.ndim == 1, "For multi-class classification, y should be of shape [B] (one-hot not supported)"
                y = y.long() # multi-class→クラスインデックス(long)を渡す
                num_classes = self.num_classes
            else:
                # binary-class
                task = "binary"
                probs = torch.sigmoid(logits)
                preds_for_metrics = probs # binary→確率を渡す
                if y.ndim == 1:
                    y = y.unsqueeze(-1)
                assert y.ndim == 2, "For binary classification, y should be of shape [B, 1]"
                y = y.long() # binary→クラスインデックス(long)を渡す
                num_classes = 1
        
            # 2. 評価指標の計算 (torchmetrics.functional を使用)
            # Accuracy
            metrics['accuracy'] = accuracy(
                preds_for_metrics, y, task=task, num_classes=num_classes
            )
        
            # Precision (適合率)
            # 多クラス分類では、通常 'macro' 平均が利用されます。
            metrics['precision'] = precision(
                preds_for_metrics, y, task=task, num_classes=num_classes, average='macro'
            )
        
            # Recall (再現率)
            metrics['recall'] = recall(
                preds_for_metrics, y, task=task, num_classes=num_classes, average='macro'
            )
        
            # F1-score (F1スコア)
            metrics['f1-score'] = f1_score(
                preds_for_metrics, y, task=task, num_classes=num_classes, average='macro'
            )
        
            # 参考: AUC (二値・多クラス対応)
            try:
                metrics['auroc'] = auroc(
                    preds_for_metrics, y, task=task, num_classes=num_classes
                )
            except:
                 # AUROCは一部のケースで計算できない場合があるため
                metrics['auroc'] = torch.tensor(float('nan'))

            return metrics

        # ----------------------------
        # Lightning hooks
        # ----------------------------
        def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
            """
            batch 仕様:
                X_total: [N_total, embed_dim]
                N_sizes: [B]
                Y_labels: [B] or [B, 1]
            """
            X_total, N_sizes, Y_labels = batch

            out = self(X_total=X_total, N_sizes=N_sizes)
            logits = out["logits"]

            loss = self._compute_loss(logits=logits, y=Y_labels)
            metrics = self._compute_metrics(logits=logits, y=Y_labels)

            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_accuracy", metrics["accuracy"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_precision", metrics["precision"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_recall", metrics["recall"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_f1-score", metrics["f1-score"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_auroc", metrics["auroc"], prog_bar=True, on_step=True, on_epoch=True)

            return loss

        def validation_step(self, batch: Tuple, batch_idx: int) -> None:
            X_total, N_sizes, Y_labels = batch

            out = self(X_total=X_total, N_sizes=N_sizes)
            logits = out["logits"]

            loss = self._compute_loss(logits=logits, y=Y_labels)
            metrics = self._compute_metrics(logits=logits, y=Y_labels)

            self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("val_accuracy", metrics["accuracy"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("val_precision", metrics["precision"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("val_recall", metrics["recall"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("val_auroc", metrics["auroc"], prog_bar=True, on_step=True, on_epoch=True)

        def test_step(self, batch: Tuple, batch_idx: int) -> None:
            X_total, N_sizes, Y_labels = batch

            out = self(X_total=X_total, N_sizes=N_sizes)
            logits = out["logits"]

            loss = self._compute_loss(logits=logits, y=Y_labels)
            metrics = self._compute_metrics(logits=logits, y=Y_labels)

            self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("test_accuracy", metrics["accuracy"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("test_precision", metrics["precision"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("test_recall", metrics["recall"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("test_f1-score", metrics["f1-score"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("test_auroc", metrics["auroc"], prog_bar=True, on_step=True, on_epoch=True)

        def predict_step(self, batch: Tuple, batch_idx: int, dataloader_idx=0) -> None:
            X_total, N_sizes = batch
            out = self(X_total=X_total, N_sizes=N_sizes)
            probs = out["probs"]
            if self.num_classes > 2:
                preds = torch.argmax(probs, dim=-1)
            else:
                preds = (probs >= 0.5).long() # 二値分類の閾値は0.5とする
            return preds

    def configure_optimizers(self) -> Dict[str, Any]:
            # 1. オプティマイザの定義 (AdamW)
            optimizer = AdamW(
                self.parameters(), 
                lr=self.hparams.lr, 
                eps=self.hparams.eps # AdamWでよく使われる小さなepsilon
            )
        
            # 2. スケジューラの定義 (Warmup付き Cosine Annealing)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.max_steps,
                num_cycles=0.5, #1.0にするとrestartが入る
            )
        
            # 3. PyTorch Lightningへの返却形式
            # 'scheduler'キーと、スケジューラの動作設定を返します
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step", # バッチごとにscheduler.step()
                    "frequency": 1, # 毎ステップ実行(2にすると2stepに1回)
                },
            }
    return (AttentionMILLitModule,)


@app.cell
def _(mo):
    mo.md(r"""
    # Training
    """)
    return


@app.cell
def _(
    AttentionMILLitModule,
    EarlyStopping,
    LearningRateMonitor,
    MILDataModule,
    ModelCheckpoint,
    TensorBoardLogger,
    Trainer,
    test_torch_dataset,
    torch,
    train_torch_dataset,
    valid_torch_dataset,
):
    # データとモデル
    data_module = MILDataModule(train_torch_dataset, valid_torch_dataset, test_torch_dataset, batch_size=4, num_workers=2)
    data_module.setup("fit")
    model = AttentionMILLitModule(
        num_classes=2,
        embed_dim=2048,
        attn_dim=256,
        dropout=0,
        lr=1e-4,
        pos_weight=1.0,  # クラス不均衡がある場合に調整
        warmup_steps=500,
        max_steps=5000,
    )

    # ====== コールバックとロガー設定 ======
    # ① モデルの自動保存
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",       # 監視対象（validation loss）
        mode="min",              # 小さい方が良い
        save_top_k=1,            # 最良モデル1つだけ保存
        filename="AttentionMIL-{epoch:02d}-{val_loss:.2f}"
    )

    # ② 学習率モニタリング
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # ③ 早期終了（EarlyStopping）
    early_stop_callback = EarlyStopping(
        monitor="val_loss",      # 監視対象（val_loss）
        mode="min",              # 小さい方が良い
        patience=3,              # 3エポック連続で改善がなければ終了
        verbose=True
    )

    # ④ TensorBoard ロガー
    logger = TensorBoardLogger(
        save_dir="lightning_logs",   # ログ保存ディレクトリ
        name="AttentionMIL"      # 実験名（サブフォルダ名）
    )

    # ====== トレーナー設定 ======
    trainer = Trainer(
        max_epochs=20,
        accelerator="auto",    # GPUが使えるなら自動でGPU
        devices=1 if torch.cuda.is_available() else None,
        precision=16 if torch.cuda.is_available() else 32,  # GPUならFP16高速化
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        logger=logger,         # TensorBoard ロガーを有効化
    )

    # ====== 学習実行 ======
    trainer.fit(model, datamodule=data_module)
    return (data_module,)


if __name__ == "__main__":
    app.run()
