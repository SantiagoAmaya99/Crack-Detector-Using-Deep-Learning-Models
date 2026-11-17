# 🏗️ Detección y Localización de Grietas en Concreto mediante Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Transfer_Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Grad--CAM-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

> **Proyecto de Inteligencia Artificial II — Universidad Industrial de Santander (UIS)**  
> **Autores:**  Santiago Amaya - Andrés Mantilla · Andrés García 

Sistema de visión por computadora diseñado para la **clasificación automática** y **localización espacial** de grietas superficiales en estructuras de concreto (cubiertas, pavimentos y muros), combinando arquitecturas convolucionales ligeras y densas (*MobileNetV2* y *DenseNet121*) con mapas de activación **Grad-CAM** refinados morfológicamente.

---

## 📌 Descripción General

La inspección visual de estructuras de concreto es costosa y propensa a errores humanos ante fisuras delgadas o condiciones variables de iluminación. Este proyecto aborda el problema en dos niveles:

1. **Clasificación binaria (`con grieta` / `sin grieta`):** Evaluación comparativa entre **MobileNetV2** (2.3 M parámetros) y **DenseNet121** (7.0 M parámetros) entrenadas bajo una estrategia de *Transfer Learning* en dos fases.
2. **Localización visual sin anotaciones de entrenamiento:** Generación de mapas de calor mediante **Grad-CAM**, combinados con detección morfológica de crestas oscuras para delimitar la trayectoria exacta de la grieta.

---

## 🧪 Datos y Metodología de Entrenamiento

### 1. Conjunto de Datos Principal (SDNET2018)
* **56,092 imágenes** de $256 \times 256$ píxeles con un desbalance natural de **15.1 % con grieta** (84.9 % sin grieta).
* **División sin fuga de datos (*Zero Data Leakage*):** Los recortes se agruparon por fotografía original (230 fotos en total) en una partición aproximada **70 / 15 / 15**:
  * **Entrenamiento:** 38,995 imágenes (160 fotos)
  * **Validación:** 8,531 imágenes (35 fotos)
  * **Prueba (Test):** 8,566 imágenes (35 fotos)

### 2. Estrategia de Entrenamiento en Dos Fases
Ambos modelos se entrenaron bajo exactamente la misma receta con optimizador **Adam**, pérdida de entropía cruzada binaria, aumento de datos exclusivo en entrenamiento (giros, rotación, brillo y contraste) y ponderación de clases (`0.59` sin grieta / `3.29` con grieta):
* **Fase 1 — Cabeza de clasificación (5 épocas, `lr = 1e-3`):** Base preentrenada en ImageNet congelada; entrenamiento exclusivo de *Global Average Pooling*, `Dropout(0.3)` y salida sigmoide.
* **Fase 2 — *Fine-tuning* (10 épocas, `lr = 1e-5`):** Descongelamiento de los bloques profundos (25 de 154 capas en MobileNetV2; 81 de 427 en DenseNet121) manteniendo `BatchNorm` congelada y conservando los pesos con mejor **PR-AUC** en validación.

---

## 📊 Resultados y Validación

### 1. Rendimiento en Prueba Interna (SDNET2018 · 8,566 imágenes)
Con solo la cabeza entrenada lidera MobileNetV2, pero tras el *fine-tuning* **DenseNet121** lo supera ligeramente en métricas globales:

| Métrica | MobileNetV2 | DenseNet121 |
| :--- | :---: | :---: |
| **Parámetros** | 2.3 M | 7.0 M |
| **AUC-ROC** | 0.926 | **0.935** |
| **PR-AUC** | 0.847 | **0.849** |
| **Precisión** *(umbral 0.5)* | 0.697 | **0.736** |
| **Recall** *(umbral 0.5)* | **0.793** | 0.778 |
| **F1-Score** *(umbral 0.5)* | 0.742 | **0.757** |

---

### 2. Generalización Externa (*Concrete Crack Images for Classification*)
Evaluación sobre una muestra balanceada de **5,000 imágenes inéditas** (2,500 con grieta y 2,500 sin grieta) proveniente de un dominio distinto:

| Métrica | MobileNetV2 *(umbral 0.5 + TTA)* | DenseNet121 *(umbral 0.38, sin TTA)* |
| :--- | :---: | :---: |
| **AUC-ROC** | 0.942 | **0.986** |
| **PR-AUC** | 0.951 | **0.982** |
| **Accuracy** | 0.887 | **0.953** |
| **Recall** | 0.830 (2,075 / 2,500) | **0.960** (2,401 / 2,500) |
| **Precisión** | 0.936 | **0.947** |
| **Especificidad** | 0.943 | **0.946** |
| **F1-Score** | 0.880 | **0.954** |

> **Lectura:** **DenseNet121** demuestra una capacidad de generalización sustancialmente superior fuera de SDNET2018 (**AUC-ROC = 0.986**), reduciendo las grietas omitidas de 425 a apenas 99.

---

### 3. Localización Espacial (50 Máscaras Anotadas Manualmente)
Evaluación de los mapas Grad-CAM refinados con crestas morfológicas frente a anotaciones manuales a nivel de píxel:

| Métrica de Localización | MobileNetV2 | DenseNet121 |
| :--- | :---: | :---: |
| **Pointing Game** *(punto máximo sobre grieta)* | **96.0 %** | 86.0 % |
| **IoU** | **0.416** | 0.309 |
| **Dice** | **0.574** | 0.454 |
| **Cobertura** | **0.692** | 0.516 |

* **Aporte del refinamiento con crestas:** Sin el filtrado morfológico de crestas, el *Pointing Game* de DenseNet121 cae al **56 %**, evidenciando que la combinación de Grad-CAM con detección de estructuras oscuras y delgadas es indispensable para localizar la fisura.
* **Compromiso umbral–área:** Umbrales bajos (`0.2`) priorizan cobertura completa (`0.69`), mientras que umbrales altos (`0.5`) acotan la predicción con alta precisión de píxel (`0.90`, marcando apenas el `2.1 %` del área de la imagen).

---

## 💡 Conclusiones y Límites

* **Complementariedad entre arquitecturas:** **DenseNet121** es el mejor clasificador y generaliza sobresalientemente a nuevos dominios (`0.986` AUC-ROC), mientras que **MobileNetV2** ofrece mapas de activación más focalizados y precisos espacialmente (`96 %` de acierto en localización).
* **Limitaciones identificadas:** La localización pierde sensibilidad en grietas extremadamente finas o discontinuas; además, los valores moderados de IoU responden a la naturaleza filiforme de las fisuras (líneas de pocos píxeles de grosor) y a que la evaluación espacial se realizó sobre 50 casos con grieta visible.
