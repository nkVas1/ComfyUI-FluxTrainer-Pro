"""
FluxTrainer Pro - Training State Singleton
==========================================

Централизованное хранилище состояния тренировки.
Доступно из нод (для записи) и из API (для чтения).

Использование:
    from .training_state import TrainingState
    state = TrainingState.instance()
    state.update_step(step=100, loss=0.023, lr=1e-4)
"""

import time
import threading
import tomllib
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum


class TrainingStatus(str, Enum):
    """Статусы процесса тренировки"""
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    SAMPLING = "sampling"
    SAVING = "saving"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TrainingMetrics:
    """Метрики одного шага тренировки"""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass 
class SampleImage:
    """Превью-изображение из тренировки"""
    step: int = 0
    epoch: int = 0
    path: str = ""
    prompt: str = ""
    timestamp: float = field(default_factory=time.time)
    width: int = 0
    height: int = 0


class TrainingState:
    """
    Singleton состояния тренировки.
    Thread-safe для доступа из нод и API одновременно.
    """
    _instance: Optional['TrainingState'] = None
    _lock = threading.Lock()
    
    @classmethod
    def instance(cls) -> 'TrainingState':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Сброс состояния (для тестов)"""
        with cls._lock:
            cls._instance = None
    
    def __init__(self):
        # Reentrant lock: to_dict вызывает методы, которые тоже используют lock.
        # Обычный Lock здесь может приводить к дедлоку API /status.
        self._data_lock = threading.RLock()
        
        # === Статус ===
        self.status: TrainingStatus = TrainingStatus.IDLE
        self.error_message: str = ""
        
        # === Прогресс ===
        self.current_step: int = 0
        self.max_steps: int = 0
        self.current_epoch: int = 0
        self.max_epochs: int = 0
        
        # === Временные метки ===
        self.training_start_time: Optional[float] = None
        self.step_start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        
        # === История метрик ===
        self.loss_history: List[Dict[str, float]] = []
        self.lr_history: List[Dict[str, float]] = []
        self.grad_norm_history: List[Dict[str, float]] = []
        self.vram_history: List[Dict[str, float]] = []
        
        # === Moving average ===
        self._recent_losses: List[float] = []
        self._recent_step_times: List[float] = []
        
        # === Мин/Макс ===
        self.min_loss: float = float('inf')
        self.max_loss: float = 0.0
        self.best_loss_step: int = 0
        
        # === Сэмплы ===
        self.sample_images: List[SampleImage] = []
        
        # === Конфигурация ===
        self.config: Dict[str, Any] = {}
        self.model_name: str = ""
        self.dataset_info: Dict[str, Any] = {}
        
        # === WebSocket callback ===
        self._ws_callback = None
        
        # === Presets ===
        self.presets: Dict[str, Dict] = {}
    
    def set_ws_callback(self, callback):
        """Установить callback для WebSocket уведомлений"""
        self._ws_callback = callback
    
    def _emit_ws(self, event_type: str, data: dict):
        """Отправить WebSocket событие"""
        if self._ws_callback:
            try:
                self._ws_callback(event_type, data)
            except Exception:
                pass  # Не ломать тренировку из-за WS ошибок

    def _extract_dataset_info_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        dataset_info: Dict[str, Any] = {}

        if not isinstance(config, dict):
            return dataset_info

        dataset_toml = config.get("dataset_config")
        if not isinstance(dataset_toml, str) or not dataset_toml.strip():
            return dataset_info

        try:
            parsed = tomllib.loads(dataset_toml)
            datasets = parsed.get("datasets", [])
            if not datasets:
                return dataset_info

            first_dataset = datasets[0] if isinstance(datasets[0], dict) else {}
            subsets = first_dataset.get("subsets", []) if isinstance(first_dataset, dict) else []

            image_dirs = []
            class_tokens = []
            subset_repeats = []
            for subset in subsets:
                if not isinstance(subset, dict):
                    continue
                image_dir = subset.get("image_dir")
                token = subset.get("class_tokens")
                repeats = subset.get("num_repeats")
                if image_dir:
                    image_dirs.append(str(image_dir))
                if token:
                    class_tokens.append(str(token))
                if repeats is not None:
                    subset_repeats.append(repeats)

            dataset_info = {
                "datasets_count": len(datasets),
                "subsets_count": len(subsets),
                "image_dirs": image_dirs,
                "class_tokens": class_tokens,
                "num_repeats": subset_repeats,
                "resolution": first_dataset.get("resolution"),
                "batch_size": first_dataset.get("batch_size"),
                "enable_bucket": first_dataset.get("enable_bucket"),
                "min_bucket_reso": first_dataset.get("min_bucket_reso"),
                "max_bucket_reso": first_dataset.get("max_bucket_reso"),
                "bucket_no_upscale": first_dataset.get("bucket_no_upscale"),
            }
        except Exception:
            return {}

        return dataset_info
    
    # === Основные методы обновления ===
    
    def start_training(self, config: Dict[str, Any], max_steps: int, max_epochs: int, model_name: str = ""):
        """Вызывается при начале тренировки"""
        with self._data_lock:
            self.status = TrainingStatus.TRAINING
            self.config = config
            inferred_dataset_info = self._extract_dataset_info_from_config(config)
            if inferred_dataset_info:
                self.dataset_info = inferred_dataset_info
            self.max_steps = max_steps
            self.max_epochs = max_epochs
            self.model_name = model_name
            self.current_step = 0
            self.current_epoch = 0
            self.training_start_time = time.time()
            self.last_update_time = time.time()
            self.error_message = ""
            
            # Очищаем историю
            self.loss_history.clear()
            self.lr_history.clear()
            self.grad_norm_history.clear()
            self.vram_history.clear()
            self.sample_images.clear()
            self._recent_losses.clear()
            self._recent_step_times.clear()
            self.min_loss = float('inf')
            self.max_loss = 0.0
            self.best_loss_step = 0
        
        self._emit_ws("fluxtrainer.started", {
            "max_steps": max_steps,
            "max_epochs": max_epochs,
            "model": model_name,
        })

    def start_preparing(self, config: Dict[str, Any], model_name: str = ""):
        """Состояние подготовки: конфиг уже виден в dashboard, обучение ещё не началось."""
        with self._data_lock:
            self.status = TrainingStatus.PREPARING
            self.config = config
            inferred_dataset_info = self._extract_dataset_info_from_config(config)
            if inferred_dataset_info:
                self.dataset_info = inferred_dataset_info
            self.model_name = model_name
            self.error_message = ""
            self.last_update_time = time.time()

        self._emit_ws("fluxtrainer.status", {
            "status": TrainingStatus.PREPARING.value,
            "message": "Подготовка тренировки",
        })

    def set_dataset_info(self, dataset_info: Dict[str, Any]):
        with self._data_lock:
            self.dataset_info = dataset_info if isinstance(dataset_info, dict) else {}
            self.last_update_time = time.time()

        self._emit_ws("fluxtrainer.dataset", {
            "dataset_info": self.dataset_info,
        })
    
    def update_step(self, step: int, loss: float, lr: float = 0.0, 
                    grad_norm: float = 0.0, epoch: int = 0):
        """Вызывается каждый шаг тренировки"""
        now = time.time()
        
        with self._data_lock:
            # Время шага
            if self.step_start_time:
                step_time = now - self.step_start_time
                self._recent_step_times.append(step_time)
                if len(self._recent_step_times) > 100:
                    self._recent_step_times.pop(0)
            self.step_start_time = now
            
            self.current_step = step
            self.current_epoch = epoch
            self.last_update_time = now
            
            # Loss
            self.loss_history.append({"step": step, "value": loss, "t": now})
            self._recent_losses.append(loss)
            if len(self._recent_losses) > 50:
                self._recent_losses.pop(0)
            
            if loss < self.min_loss:
                self.min_loss = loss
                self.best_loss_step = step
            if loss > self.max_loss:
                self.max_loss = loss
            
            # LR
            if lr > 0:
                self.lr_history.append({"step": step, "value": lr, "t": now})
            
            # Grad norm
            if grad_norm > 0:
                self.grad_norm_history.append({"step": step, "value": grad_norm, "t": now})
            
            # VRAM
            try:
                import torch
                if torch.cuda.is_available():
                    vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    self.vram_history.append({"step": step, "used": vram_used, "total": vram_total, "t": now})
            except Exception:
                pass
        
        # Каждые 10 шагов отправляем по WS
        if step % 10 == 0 or step == 1:
            self._emit_ws("fluxtrainer.progress", {
                "step": step,
                "max_steps": self.max_steps,
                "epoch": epoch,
                "loss": round(loss, 6),
                "lr": lr,
                "eta": self.get_eta(),
            })
    
    def update_vram(self, used_gb: float, total_gb: float, ram_used_gb: float = 0.0):
        """Обновление отдельно для VRAM мониторинга"""
        with self._data_lock:
            self.vram_history.append({
                "step": self.current_step,
                "used": round(used_gb, 2),
                "total": round(total_gb, 2),
                "ram": round(ram_used_gb, 2),
                "t": time.time(),
            })
    
    def add_sample(self, step: int, path: str, prompt: str = "",
                   epoch: int = 0, width: int = 0, height: int = 0):
        """Добавить сэмпл"""
        with self._data_lock:
            self.sample_images.append(SampleImage(
                step=step, epoch=epoch, path=path,
                prompt=prompt, width=width, height=height,
            ))
        
        self._emit_ws("fluxtrainer.sample", {
            "step": step,
            "path": path,
            "prompt": prompt,
        })
    
    def set_status(self, status: TrainingStatus, message: str = ""):
        """Смена статуса"""
        with self._data_lock:
            self.status = status
            if message:
                self.error_message = message
            self.last_update_time = time.time()
        
        self._emit_ws("fluxtrainer.status", {
            "status": status.value,
            "message": message,
        })
    
    def finish_training(self, success: bool = True, message: str = ""):
        """Завершение тренировки"""
        with self._data_lock:
            self.status = TrainingStatus.COMPLETED if success else TrainingStatus.ERROR
            self.error_message = message
            self.last_update_time = time.time()
        
        self._emit_ws("fluxtrainer.finished", {
            "success": success,
            "message": message,
            "total_steps": self.current_step,
            "total_time": self.get_elapsed_time(),
            "final_loss": self.loss_history[-1]["value"] if self.loss_history else 0,
        })
    
    # === Геттеры ===
    
    def get_eta(self) -> Optional[float]:
        """Оставшееся время в секундах"""
        with self._data_lock:
            if not self._recent_step_times or self.current_step >= self.max_steps:
                return None
            avg_step_time = sum(self._recent_step_times) / len(self._recent_step_times)
            remaining_steps = self.max_steps - self.current_step
            return avg_step_time * remaining_steps
    
    def get_elapsed_time(self) -> float:
        """Прошедшее время в секундах"""
        if self.training_start_time is None:
            return 0.0
        return time.time() - self.training_start_time
    
    def get_avg_loss(self, window: int = 50) -> float:
        """Moving average loss"""
        with self._data_lock:
            if not self._recent_losses:
                return 0.0
            recent = self._recent_losses[-window:]
            return sum(recent) / len(recent)
    
    def get_steps_per_second(self) -> float:
        """Скорость тренировки"""
        with self._data_lock:
            if not self._recent_step_times:
                return 0.0
            avg = sum(self._recent_step_times) / len(self._recent_step_times)
            return 1.0 / avg if avg > 0 else 0.0
    
    def get_progress_percent(self) -> float:
        """Процент выполнения"""
        if self.max_steps <= 0:
            return 0.0
        return min(100.0, (self.current_step / self.max_steps) * 100.0)
    
    def to_dict(self) -> dict:
        """Полное состояние в виде словаря (для API)"""
        with self._data_lock:
            return {
                "status": self.status.value,
                "error": self.error_message,
                "step": self.current_step,
                "max_steps": self.max_steps,
                "epoch": self.current_epoch,
                "max_epochs": self.max_epochs,
                "progress_percent": round(self.get_progress_percent(), 1),
                "elapsed_seconds": round(self.get_elapsed_time(), 1),
                "eta_seconds": self.get_eta(),
                "steps_per_second": round(self.get_steps_per_second(), 2),
                "current_loss": self.loss_history[-1]["value"] if self.loss_history else None,
                "avg_loss": round(self.get_avg_loss(), 6),
                "min_loss": self.min_loss if self.min_loss != float('inf') else None,
                "best_loss_step": self.best_loss_step,
                "model_name": self.model_name,
                "config": self.config,
                "loss_count": len(self.loss_history),
                "sample_count": len(self.sample_images),
            }
    
    def get_loss_data(self, since_step: int = 0, max_points: int = 1000) -> List[dict]:
        """Данные loss для графика (с decimate для производительности)"""
        with self._data_lock:
            data = [p for p in self.loss_history if p["step"] >= since_step]
            
            # Decimate если слишком много точек
            if len(data) > max_points:
                step = len(data) // max_points
                data = data[::step]
            
            return data
    
    def get_lr_data(self, since_step: int = 0) -> List[dict]:
        """Данные learning rate для графика"""
        with self._data_lock:
            return [p for p in self.lr_history if p["step"] >= since_step]
    
    def get_vram_data(self, last_n: int = 100) -> List[dict]:
        """Данные VRAM для графика"""
        with self._data_lock:
            return self.vram_history[-last_n:]
    
    def get_samples(self, last_n: int = 20) -> List[dict]:
        """Последние сэмплы"""
        with self._data_lock:
            samples = self.sample_images[-last_n:]
            return [
                {
                    "step": s.step,
                    "epoch": s.epoch,
                    "path": s.path,
                    "prompt": s.prompt,
                    "width": s.width,
                    "height": s.height,
                }
                for s in samples
            ]
