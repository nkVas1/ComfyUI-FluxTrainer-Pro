/**
 * ComfyUI-FluxTrainer-Pro Web Extensions
 * ======================================
 * 
 * Расширения для улучшенного UI нод FluxTrainer-Pro.
 * 
 * @author nkVas1
 * @version 2.0.0
 */

import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "FluxTrainer-Pro";

app.registerExtension({
    name: "Comfy.FluxTrainerPro",
    
    async setup() {
        console.log(`[${EXTENSION_NAME}] Web extensions loaded`);
    },
    
    /**
     * Добавляет цветовую маркировку для нод FluxTrainer
     */
    async nodeCreated(node) {
        // Помечаем Flux.2 ноды специальным цветом
        if (node.comfyClass?.startsWith("Flux2")) {
            node.bgcolor = "#1a3d2a";  // Тёмно-зелёный для Flux.2
            node.color = "#2d5a3f";
        }
        // Помечаем Legacy Flux ноды
        else if (node.comfyClass?.includes("FluxTrain") || node.comfyClass?.includes("InitFlux")) {
            node.bgcolor = "#2a2a3d";  // Тёмно-синий для Legacy Flux
            node.color = "#3f3f5a";
        }
        // Utility ноды
        else if (node.comfyClass?.includes("Dataset") || node.comfyClass?.includes("Optimizer")) {
            node.bgcolor = "#3d2a2a";  // Тёмно-красный для утилит
            node.color = "#5a3f3f";
        }
    },
    
    /**
     * Добавляет контекстное меню для быстрых действий
     */
    async getCustomWidgets() {
        return {};
    }
});

// Добавляем tooltip для LoRA path
app.registerExtension({
    name: "Comfy.FluxTrainerPro.Tooltips",
    
    async nodeCreated(node) {
        // Для нод выбора модели добавляем подсказки
        if (node.comfyClass === "Flux2TrainModelPaths" || node.comfyClass === "FluxTrainModelSelect") {
            const pathWidget = node.widgets?.find(w => w.name === "transformer" || w.name === "transformer_path");
            if (pathWidget) {
                pathWidget.tooltip = "Полный путь к файлу transformer модели (.safetensors)";
            }
        }
    }
});

console.log("[FluxTrainer-Pro] ✅ UI extensions registered");
