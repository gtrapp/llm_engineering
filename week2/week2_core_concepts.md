# LLM Engineering — Week 2 Study Notes
**Course:** AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents (Ed Donner / Udemy)
**Repo:** https://github.com/ed-donner/llm_engineering/tree/main/week2

---

## Day 1 — Multi-Provider API Comparison
**File:** `week2/day1.ipynb`

Call all four frontier providers with the same prompt and observe differences in API structure, output quality, and model behavior.

**Provider initialization:**
- OpenAI: `OpenAI()` — reads `OPENAI_API_KEY` from env
- Anthropic: `anthropic.Anthropic()` — reads `ANTHROPIC_API_KEY`
- Gemini native: `genai.configure(api_key=...)` then `genai.GenerativeModel(...)`
- Gemini via OpenAI client: `OpenAI(api_key=..., base_url="https://generativelanguage.googleapis.com/v1beta/openai/")`

**Key API differences across providers:**

| Provider | System prompt location | Response access |
|---|---|---|
| OpenAI | `messages` array, `role="system"` | `response.choices[0].message.content` |
| Anthropic | Top-level `system=` param | `response.content[0].text` |
| Gemini native | `system_instruction=` param | `response.text` |
| Gemini via OpenAI | `messages` array, `role="system"` | `response.choices[0].message.content` |

**Multi-model conversation exercise:** GPT and Claude take turns responding to each other, with each model's output passed as conversation history into the next call. Demonstrates the mechanics of history passing without a UI.

---

## Day 2 — Gradio Chatbot UI
**File:** `week2/day2.ipynb`

Wraps the multi-provider API calls in `gr.ChatInterface` — the upgrade from Week 1's one-shot `gr.Interface`.

**`gr.ChatInterface`** manages conversation history display, input clearing, and passes `history` automatically into your `chat()` function as a list of `[user_msg, assistant_msg]` pairs.

**History → messages conversion** (required on every call):
```python
def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, asst_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant",  "content": asst_msg})
    messages.append({"role": "user", "content": message})
```

**Streaming** uses `yield` — Gradio re-renders the output on every yielded value:
```python
reply = ""
for chunk in stream:
    reply += chunk.choices[0].delta.content or ""
    yield reply
```

**Provider switching** via `gr.Dropdown` as an `additional_inputs` argument — the `chat()` function dispatches to `stream_gpt()` or `stream_claude()` based on the selection.

---

## Day 3 — Conversation State & Dynamic System Prompts
**File:** `week2/day3.ipynb`

**LLMs are stateless.** There is no memory between API calls. The illusion of memory is created entirely by re-sending the full conversation history in the `messages` array on every turn:

```
Turn 1: [system, user_1]
Turn 2: [system, user_1, assistant_1, user_2]
Turn 3: [system, user_1, assistant_1, user_2, assistant_2, user_3]
```

**Token budget:** history grows with every turn and eventually hits the model's context window limit. Strategies: truncate old turns, summarize prior context, or use a sliding window over recent turns only.

**Dynamic system prompts:** the system prompt can be constructed at runtime based on conversation state — enabling persona shifts, upsell logic, or context-aware behavior changes mid-conversation:
```python
def build_system_prompt(history):
    base = "You are a helpful assistant."
    if len(history) > 5:
        base += " Be more concise now."
    return base
```

---

## Day 4 — Tool Calling
**File:** `week2/day4.ipynb`

The most important new pattern in Week 2. Tool calling lets the LLM request execution of an external Python function — giving it access to real-time data, databases, or any action your code can perform.

**The LLM never runs your code directly.** It returns a structured JSON request naming the function and arguments. Your code executes the function and sends the result back. The LLM then generates the final reply incorporating that result.

**Three components required:**

1. **The Python function** — the actual logic that runs:
```python
def get_ticket_price(destination_city):
    return ticket_prices.get(destination_city.lower(), "Unknown")
```

2. **The tool schema** — a JSON description the LLM reads to know when and how to call the tool. The `description` field is critical; the LLM uses it to decide whether to invoke the tool:
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_ticket_price",
        "description": "Get the price of a return ticket to a destination city.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination_city": {"type": "string"}
            },
            "required": ["destination_city"]
        }
    }
}]
```

3. **The two-phase `chat()` function** — first call may return a tool request; second call returns the final reply:
```python
response = openai_client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools
)
if response.choices[0].finish_reason == "tool_calls":
    # execute the tool, append result to messages
    messages.append(response.choices[0].message)
    messages.append(handle_tool_call(response.choices[0].message))
    # second call to get the final text response
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
return response.choices[0].message.content
```

**`handle_tool_call()`** parses the LLM's arguments, calls the function, and returns the result in the expected `role: "tool"` message format with a matching `tool_call_id`.

**Why it matters:** foundation for agentic systems; gives the LLM agency over *when* to fetch data or take action, not just *what* to say.

---

## Day 5 — Multi-Modal: Images & Audio
**File:** `week2/day5.ipynb`

Extends the Day 4 airline chatbot with two additional capabilities and a richer Gradio UI.

**Multiple tools:** the `tools` list grows to include both `get_ticket_price` and `book_ticket`. `handle_tool_call()` dispatches by function name:
```python
name = tool_call.function.name
if name == "get_ticket_price": ...
elif name == "book_ticket": ...
```

**Image generation (DALL-E):**
```python
response = openai_client.images.generate(
    model="dall-e-3",
    prompt=f"A travel photograph of {city}",
    size="1024x1024", n=1
)
image_url = response.data[0].url
```

**Text-to-speech (TTS):**
```python
response = openai_client.audio.speech.create(
    model="tts-1", voice="alloy", input=reply_text
)
```

**`gr.Blocks`** replaces `gr.ChatInterface` when you need multiple output panels (chat + image + audio). It gives you explicit layout control with `gr.Row()`, `gr.Column()`, and manual event wiring via `component.submit(fn=..., inputs=[...], outputs=[...])`. The `chat()` function returns a tuple — one value per output component.

---

## Week 2 Core Concepts

| Day | Key concept |
|---|---|
| Day 1 | All major providers share the same message format; only system prompt location and response access differ |
| Day 2 | `gr.ChatInterface` handles history display; you must manually rebuild the `messages` array from history on every call |
| Day 3 | LLMs are stateless — history is resent in full each turn; context window is a hard limit |
| Day 4 | Tool calling is a two-phase API call; the LLM decides *when* to call a tool, your code decides *how* |
| Day 5 | Multi-modal outputs (image, audio) require `gr.Blocks` for layout; `chat()` returns a tuple of values |

---

*Notes compiled from `week2/day1.ipynb` through `week2/day5.ipynb` in the ed-donner/llm_engineering repo.*
