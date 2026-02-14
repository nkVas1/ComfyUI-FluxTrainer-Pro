"""
FluxTrainer Pro - Training Server API
======================================

HTTP API + WebSocket endpoints для реального времени.
Подключается к ComfyUI PromptServer.

Endpoints:
    GET  /api/fluxtrainer/status       — Текущее состояние тренировки
    GET  /api/fluxtrainer/loss         — История loss для графика
    GET  /api/fluxtrainer/lr           — История learning rate
    GET  /api/fluxtrainer/vram         — История VRAM
    GET  /api/fluxtrainer/samples      — Превью-изображения
    GET  /api/fluxtrainer/config       — Текущая конфигурация
    POST /api/fluxtrainer/presets/save — Сохранить пресет
    GET  /api/fluxtrainer/presets/list — Список пресетов
    POST /api/fluxtrainer/presets/load — Загрузить пресет

WebSocket events (push):
    fluxtrainer.progress  — Прогресс каждые N шагов
    fluxtrainer.status    — Смена статуса
    fluxtrainer.sample    — Новый сэмпл
    fluxtrainer.started   — Начало тренировки
    fluxtrainer.finished  — Конец тренировки
"""

import json
import os
import logging
import traceback

logger = logging.getLogger("ComfyUI-FluxTrainer-Pro.API")

# Глобальная flag: API инициализирован
_api_initialized = False


def setup_api_routes():
    """
    Регистрирует API маршруты в ComfyUI PromptServer.
    Вызывается из __init__.py после загрузки нод.
    """
    global _api_initialized
    if _api_initialized:
        return
    
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        logger.warning("PromptServer not available — API routes not registered")
        return
    
    from .training_state import TrainingState
    
    server = PromptServer.instance
    state = TrainingState.instance()
    
    # Устанавливаем WebSocket callback
    def ws_callback(event_type: str, data: dict):
        """Отправка событий через ComfyUI WebSocket"""
        try:
            server.send_sync(event_type, data)
        except Exception as e:
            logger.debug(f"WS send failed: {e}")
    
    state.set_ws_callback(ws_callback)
    
    # === API Endpoints ===
    
    @server.routes.get("/api/fluxtrainer/status")
    async def api_status(request):
        """Полное состояние тренировки"""
        try:
            return web.json_response(state.to_dict())
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    @server.routes.get("/api/fluxtrainer/loss")
    async def api_loss(request):
        """История loss для графика"""
        try:
            since_step = int(request.query.get("since", 0))
            max_points = int(request.query.get("max_points", 2000))
            data = state.get_loss_data(since_step=since_step, max_points=max_points)
            return web.json_response({"data": data, "count": len(data)})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    @server.routes.get("/api/fluxtrainer/lr")
    async def api_lr(request):
        """История learning rate"""
        try:
            since_step = int(request.query.get("since", 0))
            data = state.get_lr_data(since_step=since_step)
            return web.json_response({"data": data, "count": len(data)})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    @server.routes.get("/api/fluxtrainer/vram")
    async def api_vram(request):
        """История VRAM"""
        try:
            last_n = int(request.query.get("last", 100))
            data = state.get_vram_data(last_n=last_n)
            return web.json_response({"data": data, "count": len(data)})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    @server.routes.get("/api/fluxtrainer/samples")
    async def api_samples(request):
        """Превью-изображения"""
        try:
            last_n = int(request.query.get("last", 20))
            data = state.get_samples(last_n=last_n)
            return web.json_response({"data": data, "count": len(data)})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    @server.routes.get("/api/fluxtrainer/config")
    async def api_config(request):
        """Текущая конфигурация тренировки"""
        try:
            return web.json_response({
                "config": state.config,
                "model_name": state.model_name,
                "dataset_info": state.dataset_info,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @server.routes.get("/api/fluxtrainer/dataset")
    async def api_dataset(request):
        """Информация по датасету текущей сессии"""
        try:
            return web.json_response({
                "dataset_info": state.dataset_info,
                "model_name": state.model_name,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    # === Presets ===
    
    _presets_dir = os.path.join(os.path.dirname(__file__), "presets")
    os.makedirs(_presets_dir, exist_ok=True)
    
    @server.routes.get("/api/fluxtrainer/presets/list")
    async def api_presets_list(request):
        """Список сохранённых пресетов"""
        try:
            presets = []

            if state.config:
                presets.append({
                    "name": "Текущая сессия",
                    "filename": "__current_session__.json",
                    "description": "Runtime preset из активной конфигурации",
                    "model_type": state.config.get("model_type", "flux2"),
                    "created": "runtime",
                    "runtime": True,
                })

            for f in os.listdir(_presets_dir):
                if f.endswith('.json'):
                    fpath = os.path.join(_presets_dir, f)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as fp:
                            data = json.load(fp)
                            presets.append({
                                "name": data.get("name", f.replace('.json', '')),
                                "filename": f,
                                "description": data.get("description", ""),
                                "model_type": data.get("model_type", "flux"),
                                "created": data.get("created", ""),
                            })
                    except Exception:
                        pass
            return web.json_response({"presets": presets})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    @server.routes.post("/api/fluxtrainer/presets/save")
    async def api_presets_save(request):
        """Сохранить пресет"""
        try:
            data = await request.json()
            name = data.get("name", "unnamed")
            filename = name.lower().replace(' ', '_').replace('/', '_') + '.json'
            filepath = os.path.join(_presets_dir, filename)
            
            import time
            preset_data = {
                "name": name,
                "description": data.get("description", ""),
                "model_type": data.get("model_type", "flux"),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": data.get("config", {}),
            }
            
            with open(filepath, 'w', encoding='utf-8') as fp:
                json.dump(preset_data, fp, indent=2, ensure_ascii=False)
            
            return web.json_response({"success": True, "filename": filename})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    @server.routes.post("/api/fluxtrainer/presets/load")
    async def api_presets_load(request):
        """Загрузить пресет"""
        try:
            data = await request.json()
            filename = data.get("filename", "")

            if filename == "__current_session__.json":
                return web.json_response({
                    "success": True,
                    "preset": {
                        "name": "Текущая сессия",
                        "description": "Runtime preset из активной конфигурации",
                        "model_type": state.config.get("model_type", "flux2"),
                        "config": state.config,
                    },
                })

            filepath = os.path.join(_presets_dir, filename)
            
            if not os.path.exists(filepath):
                return web.json_response({"error": "Preset not found"}, status=404)
            
            with open(filepath, 'r', encoding='utf-8') as fp:
                preset_data = json.load(fp)
            
            return web.json_response({"success": True, "preset": preset_data})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    # === Статическая раздача sample images ===
    @server.routes.get("/api/fluxtrainer/sample_image/{filename}")
    async def api_sample_image(request):
        """Получить sample image по имени файла"""
        try:
            filename = request.match_info.get('filename', '')
            # Ищем в sample_images по имени файла
            for sample in state.sample_images:
                if os.path.basename(sample.path) == filename:
                    if os.path.exists(sample.path):
                        return web.FileResponse(sample.path)
            return web.json_response({"error": "Not found"}, status=404)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    _api_initialized = True
    logger.info("Dashboard API routes registered (/api/fluxtrainer/*)")


def is_api_ready() -> bool:
    """Проверка инициализации API"""
    return _api_initialized
