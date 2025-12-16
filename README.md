
# LLaVA-HTML-Coder: Image-to-HTML Generation with Unsloth PEFT

## 💡 Project Overview

This project successfully fine-tunes the **LLaVA-1.5 7B** Vision-Language Model (VLM) for the specialized task of **Image-to-HTML Code Generation**. The goal is to take an image of a webpage or UI and generate the corresponding HTML code. We leverage **Unsloth** for Parameter-Efficient Fine-Tuning (PEFT) using QLoRA, enabling faster training and reduced memory consumption, making it feasible on consumer-grade GPUs like the T4. The model is trained on a small subset of the multi-modal **HuggingFaceM4/WebSight** dataset.

## ✨ Key Features (Technical Stack)

This notebook showcases an efficient and state-of-the-art approach to VLM fine-tuning:

  * **Base Model:** Uses the robust **LLaVA-1.5 7B** VLM, which is capable of complex visual instruction following.
  * **PEFT Optimization:** Implements **Unsloth's QLoRA** integration for highly memory-efficient fine-tuning, dramatically reducing training time and VRAM usage.
  * **Target Task:** Visual-to-Code generation, specifically converting an image of a webpage into raw HTML.
  * **Training Data:** Fine-tuned on a small, curated subset of the **WebSight** dataset, containing image-to-code pairs.
  * **Evaluation:** Includes a basic evaluation script using the **Corpus BLEU Score** for text similarity comparison between generated and ground-truth code.

## ⚙️ Setup and Requirements

This notebook is designed to run in an environment with a CUDA-enabled GPU.

### Hardware

Training was tested and optimized for environments with at least **16GB VRAM** (e.g., NVIDIA T4, V100, or A100), though Unsloth optimizations might allow it to run on smaller cards.

### Installation

Run the initial cell blocks in the notebooks to install the packages.


## 📂 Data Preparation

The notebook uses the `HuggingFaceM4/WebSight` dataset, which contains images of webpages and their corresponding HTML code.

The key data preparation step is converting the image and HTML pair into the format expected by the LLaVA model: the **LLaVA Conversation Template**.

1.  **Data Source:** A streaming subset of the WebSight dataset is loaded (e.g., the first 101 samples).
2.  **Formatting:** The `convert_to_conversation` function structures each sample into a multi-turn chat format:
    ```
    USER: <image> Generate the HTML code for this webpage.
    ASSISTANT: [HTML CODE HERE]
    ```
    This format is essential for the multi-modal instruction-following capability of LLaVA.
3.  **Splitting:** The data is split into a very small **training set (approx. 80 samples)** and a **validation set (approx. 21 samples)** for this rapid proof-of-concept run.

## 🚀 Fine-Tuning & Configuration

The model is fine-tuned using Unsloth's implementation of LoRA, which targets both the vision and language components of the VLM.

### Model and LoRA Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model** | `llava-1.5-7b-hf` | Base LLaVA checkpoint. |
| **Load in 4-bit** | `True` | Uses QLoRA for memory reduction. |
| **LoRA `r`** | 16 | Rank of the LoRA matrices. |
| **LoRA `alpha`** | 16 | Scaling factor for LoRA. |
| **Finetune Vision** | `True` | **Crucial:** Enables training of the vision projection layers. |
| **LoRA Target** | All default attention and projection layers. | Targets both LLaVA's CLIP and LLM components. |

### Training Arguments

The training configuration is set up for a fast demo run:

| Parameter | Value | Note |
| :--- | :--- | :--- |
| **`max_steps`** | 30 | Set intentionally low for rapid verification (smoke test). |
| **Batch Size** | 8 (Total) | Achieved via `per_device_train_batch_size=4` and `gradient_accumulation_steps=2`. |
| **Optimizer** | `paged_adamw_8bit` | Optimized for low-memory environments. |
| **Learning Rate** | `2e-5` | Standard fine-tuning learning rate. |

## 📊 Evaluation and Results

The model's performance on the validation set was measured using the Corpus BLEU score.

The final **Corpus BLEU Score** achieved after 30 steps of fine-tuning is:

$$\text{Corpus BLEU Score: } \mathbf{25.738}$$

### Interpretation

  * BLEU is a *text similarity* metric and generally serves as a quick proxy for code generation quality. A score of 25.7 is good for a very short fine-tuning run on a small dataset, indicating significant word and phrase overlap between the generated HTML and the ground truth.
  * **Next Steps:** For a robust code-generation benchmark, future improvements should include metrics that measure the structural correctness of the HTML (e.g., Tree-Edit Distance or DOM Node Accuracy), as pure text overlap can be misleading for code.
