# LLM Engineering — Week 3 Study Notes
**Course:** AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents (Ed Donner / Udemy)
**Repo:** https://github.com/ed-donner/llm_engineering/tree/main/week3

> Week 3 shifts entirely to the open-source ecosystem. All notebooks run in **Google Colab** (GPU required), not locally. Colab links are embedded inside each day's notebook. All five notebooks are present: `day1.ipynb` through `day5.ipynb`.

---

## Day 1 — Hugging Face Hub & Google Colab Setup
**File:** `week3/day1.ipynb`

**Hugging Face** is the central hub for open-source AI — hosting 800k+ models, 200k+ datasets, and deployable Spaces (apps built with Gradio or Streamlit). It is the primary platform for sourcing, sharing, and running open-source models throughout the rest of this course.

**Google Colab** provides cloud-hosted Jupyter notebooks with free GPU access (T4) or paid options (A100). It eliminates local hardware requirements for running large models.

**Setup steps:**
1. Create a Hugging Face account at huggingface.co
2. Generate an access token: Avatar → Settings → Access Tokens (set Write permissions)
3. In Colab: store the token under the **Secrets** panel (not in code) and access it with `userdata.get('HF_TOKEN')`
4. Authenticate: `from huggingface_hub import login; login(token=HF_TOKEN)`
5. Verify GPU: `!nvidia-smi`

**Key concept:** Some models (like Llama) require you to accept Terms of Service on the Hugging Face model page before the token grants download access.

---

## Day 2 — HuggingFace Pipelines
**File:** `week3/day2.ipynb`

The `pipeline()` function is the highest-level API in the HuggingFace `transformers` library. It wraps model download, tokenization, inference, and decoding into a single call — the fastest way to run any supported task.

**Basic usage:**
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love working with LLMs!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

**Assigning to GPU:**
```python
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B", device="cuda")
```

**Common NLP tasks available as pipelines:**

| Task string | What it does |
|---|---|
| `"sentiment-analysis"` | Positive/negative classification |
| `"ner"` | Named entity recognition |
| `"question-answering"` | Extractive QA from a context passage |
| `"summarization"` | Abstractive text summarization |
| `"translation_en_to_fr"` | Language translation |
| `"zero-shot-classification"` | Classify text against arbitrary labels |
| `"text-generation"` | Autoregressive text generation |
| `"automatic-speech-recognition"` | Speech-to-text (e.g. Whisper) |

**Multimodal pipelines:**
- Image generation: via the `diffusers` library with Stable Diffusion models
- Text-to-speech: `pipeline("text-to-speech")` with speaker embeddings for voice customization

**Choosing a specific model:** pass `model="org/model-name"` to override the pipeline's default. Every model on the Hub has a card showing the task it supports and example code.

**Key concept:** Pipelines are great for inference but abstract away the internals. Days 3 and 4 progressively remove that abstraction to give full control.

---

## Day 3 — Tokenizers & Chat Templates
**File:** `week3/day3.ipynb`

**Tokenizers** convert raw text into token IDs that models process, and decode IDs back to text. Every model has its own tokenizer — using the wrong one silently degrades performance.

**Loading and using a tokenizer:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Encode text → token IDs
ids = tokenizer.encode("Hello, world!")

# Decode IDs → text
text = tokenizer.decode(ids)

# Inspect individual tokens
tokens = tokenizer.convert_ids_to_tokens(ids)
```

**Key tokenizer concepts:**
- **Vocabulary:** the complete set of tokens the model knows; typically 32k–128k entries
- **Special tokens:** `<|begin_of_text|>`, `<|end_of_text|>`, `<|eot_id|>` etc. — model-specific markers that structure input
- **Subword tokenization:** words are split into pieces; "tokenization" might be 3–4 tokens
- **Token-to-character ratio:** roughly 1 token ≈ 4 characters in English; varies by language and model

**Chat templates** format a conversation history into the exact string structure an instruct-tuned model expects. Without this, the model won't behave as a chat assistant:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is 2 + 2?"}
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,          # return the string, not token IDs
    add_generation_prompt=True  # append the assistant turn opener
)
```

Each model family has a unique template format. Llama 3, Phi-3, and Qwen2 all produce different formatted strings from the same messages list — always use `apply_chat_template()` rather than manually constructing the string.

---

## Day 4 — Direct Model Inference & Quantization
**File:** `week3/day4.ipynb`

Day 4 removes the pipeline abstraction entirely, loading the model class directly. This gives full control over generation parameters, memory usage, and output handling.

**Loading a model directly:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",       # automatically place layers on available GPU/CPU
    torch_dtype=torch.bfloat16  # use half precision to reduce memory
)
```

**Generating output:**
```python
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
```

**4-bit quantization with BitsAndBytes:**
Large models (7B+ parameters) may not fit in GPU memory at full precision. Quantization compresses weights from 16-bit floats to 4-bit integers, cutting memory ~4× with minimal quality loss:

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=quant_config,
    device_map="auto"
)
```

**Streaming output token-by-token** with `TextStreamer`:
```python
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
model.generate(inputs, max_new_tokens=200, streamer=streamer)
```

**Memory management:** always clean up between experiments in Colab to avoid OOM errors:
```python
del model
torch.cuda.empty_cache()
```

**Inspecting model internals:** `print(model)` shows the full PyTorch architecture — embedding layers, attention heads, MLP dimensions, and parameter counts. Useful for understanding transformer structure.

---

## Day 5 — Meeting Minutes Generator (End-to-End Project)
**File:** `week3/day5.ipynb`

The week's capstone combines Whisper (speech-to-text) and an open-source LLM (Llama 3.1 quantized) into a real-world pipeline: audio in → structured meeting minutes out.

**Step 1 — Transcribe audio with Whisper:**
```python
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",    # Whisper runs as a HF pipeline
    torch_dtype=torch.float16,
    device="cuda"
)
result = pipe("meeting.mp3", return_timestamps=True)
transcript = result["text"]
```

**Step 2 — Generate structured minutes with a quantized LLM:**
The transcript is passed as the user message. The system prompt instructs the model to extract key discussion points, decisions, and action items in structured markdown:

```python
messages = [
    {"role": "system", "content": "You are an expert meeting summarizer. Extract key points, decisions, and action items in markdown format."},
    {"role": "user",   "content": f"Please create meeting minutes from this transcript:\n\n{transcript}"}
]
# → run through the quantized Llama 3.1 model from Day 4
```

**Why this architecture matters:**
- Whisper is a specialized audio model — swap it in at the pipeline level without touching the text generation logic
- The LLM is fully local; no audio or text leaves the Colab instance
- Demonstrates chaining two completely different model types (ASR + LLM) via a shared text interface

---

## Week 3 Core Concepts

| Day | Key concept |
|---|---|
| Day 1 | Hugging Face Hub is the model registry; Colab provides the GPU; a token with correct permissions unlocks model access |
| Day 2 | `pipeline()` is the fastest path to inference — one line for any supported task; `device="cuda"` moves it to GPU |
| Day 3 | Every model needs its own tokenizer; `apply_chat_template()` formats messages into the exact string an instruct model expects |
| Day 4 | Direct `AutoModelForCausalLM` loading gives full control; 4-bit quantization (BitsAndBytes) makes 8B models fit in Colab GPU memory |
| Day 5 | Pipeline models can be chained — Whisper transcribes audio, a quantized LLM structures the text; the only interface between them is a string |

**Three levels of HuggingFace API (introduced across Days 2–4):**

| Level | API | Best for |
|---|---|---|
| High | `pipeline("task")` | Fast prototyping, standard tasks |
| Mid | `AutoTokenizer` + `pipeline` with custom model | Task customization, specific model selection |
| Low | `AutoTokenizer` + `AutoModelForCausalLM` + `model.generate()` | Full control, quantization, streaming, fine-tuning |

**Key terms:**
- **Hugging Face Hub** — repository hosting models, datasets, and Spaces
- **Colab Secrets** — secure key storage in Colab notebooks; access via `userdata.get()`
- **`pipeline()`** — high-level HuggingFace inference API; handles tokenization, inference, and decoding
- **Tokenizer** — model-specific encoder/decoder; converts text ↔ token IDs
- **Special tokens** — model-specific markers (`<|begin_of_text|>` etc.) that structure input
- **Chat template** — model-specific formatting applied by `apply_chat_template()` to prepare message lists for instruct models
- **Quantization** — compressing model weights (e.g. 16-bit → 4-bit) to reduce memory footprint
- **BitsAndBytes** — HuggingFace-integrated quantization library; enables 4-bit loading via `BitsAndBytesConfig`
- **`TextStreamer`** — streams generated tokens to output in real time during `model.generate()`
- **Whisper** — OpenAI's open-source speech recognition model; available as a HuggingFace pipeline

---

*Notes compiled from `week3/day1.ipynb` through `week3/day5.ipynb` in the ed-donner/llm_engineering repo.*
