# Microfiction Generation with GPT2

This repository hosts a quantized version of the GPT2 model, fine-tuned for microfiction generation tasks where model will generate short stories based on the prompt. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

## Model Details
- **Model Architecture:** GPT2
- **Task:** Microfiction Generation  
- **Dataset:** Hugging Face's `Children-Stories-Collection`  
- **Quantization:** Float16
- **Fine-tuning Framework:** Hugging Face Transformers  

## Usage
### Installation
```sh
pip install transformers torch
```

### Loading the Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/gpt2-microfiction-generation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def generate_story(prompt_text):
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")
    
    output = model.generate(
        input_ids,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=True,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test Example
prompt = "Write a short science fiction story about a man wakes up in an abandoned house with no memory of how he got there."
generated_text = generate_story(prompt)
print(generated_text)
```

## 📊 ROUGE Evaluation Results
After fine-tuning the T5-Small model for text translation, we obtained the following ROUGE scores:

| **Metric**  | **Score** | **Meaning**  |
|------------|---------|--------------------------------------------------------------|
| **ROUGE-1**  | 0.4673 (~46%) | Measures overlap of unigrams (single words) between the reference and generated text. |
| **ROUGE-2**  | 0.2486 (~24%) | Measures overlap of bigrams (two-word phrases), indicating coherence and fluency. |
| **ROUGE-L**  | 0.4595 (~45%) | Measures longest matching word sequences, testing sentence structure preservation. |
| **ROUGE-Lsum**  | 0.4620 (~46%) | Similar to ROUGE-L but optimized for summarization tasks. |

## Fine-Tuning Details
### Dataset
The Hugging Face's `Children-Stories-Collection` dataset was used, containing various types of prompts and their related responses.

### Training
- **Number of epochs:** 3  
- **Save Steps:** 5000  
- **Logging Steps:** 500  

### Quantization
Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

## Repository Structure
```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## Limitations
- The model may not generalize well to domains outside the fine-tuning dataset.
- Currently, it only supports English to French translations.
- Quantization may result in minor accuracy degradation compared to full-precision models.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
