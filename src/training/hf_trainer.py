# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Wrapper around :class:`transformers.Trainer` for AG News fine-tuning.

The wrapper centralises the boilerplate that every transformer fine-tuning
in the project shares:

* tokenisation with truncation and padding;
* setup of :class:`TrainingArguments` from a :class:`TrainingConfig`;
* ``compute_metrics`` callback driven by :func:`src.evaluation.metrics.compute_metrics`;
* early stopping and best-checkpoint persistence;
* prediction collection on the validation and test splits.

Returning :class:`TransformerRunResult` keeps the trainer's outputs
homogeneous with the classical baselines so the comparison code can treat
all models uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.configs import DataConfig, TrainingConfig
from src.evaluation.metrics import (
    classification_report_table,
    compute_metrics,
    confusion_matrix_table,
)
from src.utils.io_utils import ensure_dir, save_json
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


class RDropTrainer(Trainer):
    """Trainer subclass that adds R-Drop regularisation (Liang et al., 2021).

    Each training step performs two forward passes with independent dropout
    masks and minimises ``CE_loss + alpha * KL_sym``, where ``CE_loss`` is the
    average cross-entropy of the two predictions (with label smoothing if
    configured) and ``KL_sym`` is the symmetric KL divergence between the
    two predicted distributions. Setting ``alpha = 0`` recovers vanilla
    training.
    """

    def __init__(self, *args, r_drop_alpha: float = 1.0,
                 label_smoothing: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.r_drop_alpha = r_drop_alpha
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # Two independent forward passes; PyTorch dropout reseeds each call
        # when the model is in train mode, so the two outputs differ.
        outputs1 = model(**inputs)
        outputs2 = model(**inputs)
        logits1 = outputs1.logits
        logits2 = outputs2.logits

        ce_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        ce1 = ce_loss_fn(logits1.view(-1, logits1.size(-1)), labels.view(-1))
        ce2 = ce_loss_fn(logits2.view(-1, logits2.size(-1)), labels.view(-1))
        ce_loss = 0.5 * (ce1 + ce2)

        # Symmetric KL divergence between the two softmax distributions.
        log_p1 = F.log_softmax(logits1, dim=-1)
        log_p2 = F.log_softmax(logits2, dim=-1)
        p1 = log_p1.exp()
        p2 = log_p2.exp()
        kl_12 = F.kl_div(log_p1, p2, reduction="batchmean")
        kl_21 = F.kl_div(log_p2, p1, reduction="batchmean")
        kl_loss = 0.5 * (kl_12 + kl_21)

        loss = ce_loss + self.r_drop_alpha * kl_loss
        return (loss, outputs1) if return_outputs else loss


@dataclass
class TransformerRunResult:
    name: str
    metrics: dict[str, float]
    test_predictions: np.ndarray
    test_probabilities: np.ndarray
    artefact_paths: dict[str, Path]


class TransformerTrainer:
    """Fine-tune a HuggingFace classification model on AG News."""

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        data_cfg: DataConfig,
        training_cfg: TrainingConfig,
        label_names: tuple[str, ...],
        output_dir: Path | str,
        run_name: str,
        report_to: str = "none",
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.data_cfg = data_cfg
        self.training_cfg = training_cfg
        self.label_names = label_names
        self.output_dir = ensure_dir(output_dir)
        self.run_name = run_name
        self.report_to = report_to

    # ------------------------------------------------------------------ public API

    def fit_and_evaluate(
        self,
        train: Dataset,
        validation: Dataset,
        test: Dataset,
    ) -> TransformerRunResult:
        run_dir = ensure_dir(self.output_dir / self.run_name)
        tokenised = self._tokenise_splits(train, validation, test)
        trainer = self._build_trainer(
            train_ds=tokenised["train"],
            eval_ds=tokenised["validation"],
            run_dir=run_dir,
        )
        # Auto-resume from the latest checkpoint when one is already
        # present on disk (for example, after a cloud GPU instance was
        # paused mid-training and restarted). When no checkpoints exist
        # the call is equivalent to a fresh ``trainer.train()``.
        ckpt_dir = run_dir / "checkpoints"
        resume_from = (
            True
            if ckpt_dir.exists()
            and any(ckpt_dir.glob("checkpoint-*"))
            else None
        )
        trainer.train(resume_from_checkpoint=resume_from)
        trainer.save_model(str(run_dir / "best"))
        self.tokenizer.save_pretrained(str(run_dir / "best"))

        val_metrics = trainer.evaluate(eval_dataset=tokenised["validation"])
        val_metrics = {
            f"val_{k.removeprefix('eval_')}": v
            for k, v in val_metrics.items()
            if isinstance(v, (int, float))
        }

        prediction_output = trainer.predict(tokenised["test"])
        test_logits = prediction_output.predictions
        test_pred = np.argmax(test_logits, axis=1)
        test_proba = self._softmax(test_logits)
        test_metrics = compute_metrics(np.asarray(test[self.data_cfg.label_column]), test_pred)
        test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}

        all_metrics = {**val_metrics, **test_metrics_prefixed}
        save_json(all_metrics, run_dir / "metrics.json")

        report = classification_report_table(
            np.asarray(test[self.data_cfg.label_column]),
            test_pred,
            label_names=self.label_names,
        )
        report.to_csv(run_dir / "classification_report.csv", index=False)

        cm = confusion_matrix_table(
            np.asarray(test[self.data_cfg.label_column]),
            test_pred,
            label_names=self.label_names,
        )
        cm.to_csv(run_dir / "confusion_matrix.csv")

        np.save(run_dir / "test_probabilities.npy", test_proba)
        pd.DataFrame(
            {
                "text": test[self.data_cfg.text_column],
                "label": test[self.data_cfg.label_column],
                "label_name": [self.label_names[i] for i in test[self.data_cfg.label_column]],
                "prediction": test_pred,
                "prediction_name": [self.label_names[i] for i in test_pred],
                "max_probability": test_proba.max(axis=1),
            }
        ).to_csv(run_dir / "test_predictions.csv", index=False)

        _logger.info(
            "Run '%s' finished. Test accuracy=%.4f, F1-macro=%.4f",
            self.run_name,
            test_metrics["accuracy"],
            test_metrics["f1_macro"],
        )
        return TransformerRunResult(
            name=self.run_name,
            metrics=all_metrics,
            test_predictions=test_pred,
            test_probabilities=test_proba,
            artefact_paths={
                "model": run_dir / "best",
                "metrics": run_dir / "metrics.json",
                "report": run_dir / "classification_report.csv",
                "confusion_matrix": run_dir / "confusion_matrix.csv",
                "predictions": run_dir / "test_predictions.csv",
                "probabilities": run_dir / "test_probabilities.npy",
            },
        )

    # ---------------------------------------------------------------- internals

    def _tokenise_splits(
        self, train: Dataset, validation: Dataset, test: Dataset
    ) -> dict[str, Dataset]:
        text_col = self.data_cfg.text_column
        label_col = self.data_cfg.label_column

        def tokenise(batch: dict[str, list]) -> dict[str, list]:
            tokenised = self.tokenizer(
                batch[text_col],
                truncation=True,
                max_length=self.data_cfg.max_length,
            )
            tokenised["labels"] = batch[label_col]
            return tokenised

        out = {}
        for name, ds in (("train", train), ("validation", validation), ("test", test)):
            keep_cols = [text_col, label_col]
            removed = [c for c in ds.column_names if c not in keep_cols]
            out[name] = ds.map(tokenise, batched=True, remove_columns=removed)
        return out

    def _build_trainer(
        self,
        *,
        train_ds: Dataset,
        eval_ds: Dataset,
        run_dir: Path,
    ) -> Trainer:
        cfg = self.training_cfg
        args = TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            run_name=self.run_name,
            seed=cfg.seed,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            optim=cfg.optimizer,
            lr_scheduler_type=cfg.scheduler,
            # ``evaluation_strategy`` was renamed to ``eval_strategy`` in
            # transformers >= 4.40; we use the new name to stay forward
            # compatible with the latest transformers wheel.
            eval_strategy="epoch" if cfg.eval_steps is None else "steps",
            eval_steps=cfg.eval_steps,
            save_strategy="epoch" if cfg.eval_steps is None else "steps",
            save_steps=cfg.eval_steps,
            save_total_limit=cfg.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=cfg.logging_steps,
            # When R-Drop is active, the custom compute_loss applies label
            # smoothing manually inside the cross-entropy term, so the
            # built-in label_smoothing_factor must be disabled to avoid
            # double-counting.
            label_smoothing_factor=(
                0.0 if cfg.r_drop_alpha > 0 else cfg.label_smoothing_factor
            ),
            report_to=self.report_to,
            disable_tqdm=False,
        )

        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            return compute_metrics(labels, preds)

        callbacks = []
        if cfg.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
            )

        if cfg.r_drop_alpha > 0:
            return RDropTrainer(
                model=self.model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPadding(self.tokenizer),
                compute_metrics=compute_metrics_fn,
                callbacks=callbacks,
                r_drop_alpha=cfg.r_drop_alpha,
                label_smoothing=cfg.label_smoothing_factor,
            )
        return Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics_fn,
            callbacks=callbacks,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
