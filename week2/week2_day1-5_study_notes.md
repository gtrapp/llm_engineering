# LLM Engineering — Week 2 Study Notes
**Course:** AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents (Ed Donner / Udemy)
**Repo:** https://github.com/ed-donner/llm_engineering/tree/main/week2

> Week 2 builds on the Week 1 web summarizer foundation and progressively constructs a full multi-provider chatbot, then extends it with tool calling and multi-modal features. All five notebooks are present: `day1.ipynb` through `day5.ipynb`.

---

## Day 1 — Multi-Provider API Comparison
**File:** `week2/day1.ipynb`

### What this day covers
The week opens by wiring up all four major frontier providers in a single notebook and calling each one with the same prompt — making the differences in API structure, output, and model behavior directly visible.

### Initializing multiple providers

```python
from openai import OpenAI
import anthropic
import google.generativeai as genai

# OpenAI
openai_client = OpenAI()

# Anthropic
claude_client = anthropic.Anthropic()

# Google Gemini — native SDK
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Google via OpenAI-compatible endpoint
google_via_openai = OpenAI(
    api_key=os.environ["GOOGLE_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
```

### Standard message format across providers
All providers receive the same structured messages array. The differences lie in how `system` is passed and where the response content lives:

```python
# Used as-is for OpenAI and Google (via OpenAI client)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": user_prompt}
]
```

### Basic API calls — one prompt, four providers

**OpenAI:**
```python
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print(response.choices[0].message.content)
```

**Anthropic (system is top-level, not in messages):**
```python
response = claude_client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    system=system_prompt,
    messages=[{"role": "user", "content": user_prompt}]
)
print(response.content[0].text)
```

**Google Gemini (native SDK):**
```python
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=system_prompt
)
response = model.generate_content(user_prompt)
print(response.text)
```

**Google via OpenAI client:**
```python
response = google_via_openai.chat.completions.create(
    model="gemini-2.0-flash",
    messages=messages
)
print(response.choices[0].message.content)
```

### Streaming across providers
All providers are called with streaming enabled so tokens render in real time:

**OpenAI streaming:**
```python
stream = openai_client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

**Anthropic streaming:**
```python
with claude_client.messages.stream(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    system=system_prompt,
    messages=[{"role": "user", "content": user_prompt}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Multi-model "conversation" exercise
A creative exercise where GPT and Claude take turns responding to each other in a simulated back-and-forth, with their messages being passed as conversation history to the other model. This demonstrates the mechanics of conversation history without building a full chatbot UI.

### Key API parameter reference
| Parameter | Purpose | Notes |
|---|---|---|
| `model` | Which model to call | `gpt-4o-mini`, `claude-3-haiku-20240307`, `gemini-2.0-flash` |
| `messages` | Conversation context | Array of `{role, content}` objects |
| `max_tokens` | Cap on response length | Required for Anthropic; optional for others |
| `temperature` | Randomness (0–1) | Higher = more creative |
| `stream` | Incremental output | `True`/`False` |

---

## Day 2 — Gradio Chatbot UI with Multiple Providers
**File:** `week2/day2.ipynb`

### What this day covers
Day 2 wraps the multi-provider API calls from Day 1 inside a Gradio UI, building a proper interactive chatbot that lets the user choose which model answers their message.

### `gr.Interface` vs `gr.ChatInterface`
Two Gradio patterns are introduced and compared:

- `gr.Interface` — a generic function wrapper; suitable for one-shot inputs/outputs (seen in Week 1)
- `gr.ChatInterface` — purpose-built for chat; automatically manages conversation history display, input clearing, and history passing into your function

### `gr.ChatInterface` basics
```python
import gradio as gr

def chat(message, history):
    # history is a list of [user, assistant] pairs managed by Gradio
    # build messages from history + new message, call LLM, return reply
    ...
    return reply

gr.ChatInterface(fn=chat).launch()
```

### Building the messages list from Gradio history
Gradio passes history as a list of `[user_message, assistant_message]` pairs. You must convert it back into the API messages format on each call:

```python
def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant",  "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    # call LLM with messages...
```

### Streaming in `gr.ChatInterface`
Use `yield` to stream responses into the Gradio chat window:

```python
def chat(message, history):
    messages = build_messages(history, message)
    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, stream=True
    )
    reply = ""
    for chunk in stream:
        reply += chunk.choices[0].delta.content or ""
        yield reply   # Gradio updates the UI on each yield
```

### Provider dropdown — switching models in the UI
Adding a `gr.Dropdown` as an additional input lets the user select which provider responds:

```python
def chat(message, history, provider):
    if provider == "GPT":
        return stream_gpt(message, history)
    elif provider == "Claude":
        return stream_claude(message, history)

gr.ChatInterface(
    fn=chat,
    additional_inputs=[
        gr.Dropdown(["GPT", "Claude"], label="Provider", value="GPT")
    ]
).launch()
```

### `stream_gpt()` and `stream_claude()` as named functions
Streaming logic for each provider is extracted into standalone functions — this separation keeps the `chat()` dispatcher clean and each provider's idiosyncrasies isolated.

### Launch options
```python
.launch()              # local only, http://127.0.0.1:7860
.launch(share=True)    # generates a public Gradio link (72hr expiry)
.launch(inbrowser=True) # auto-opens the browser tab
```

---

## Day 3 — Conversation History and System Prompt Engineering
**File:** `week2/day3.ipynb`

### What this day covers
Day 3 deepens the chatbot's conversational behavior — focusing on how conversation history is managed across turns, and how the system prompt can be dynamically modified based on what's happening in the conversation.

### How conversation state is maintained
LLMs are stateless by design — they have no memory between calls. The illusion of memory is created by re-sending the full conversation history on every API call. Each turn appends to the history; the whole list is passed in the `messages` array:

```
Turn 1: [system, user_1]
Turn 2: [system, user_1, assistant_1, user_2]
Turn 3: [system, user_1, assistant_1, user_2, assistant_2, user_3]
```

### The `chat()` function pattern — complete version
```python
def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, asst_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant",  "content": asst_msg})
    messages.append({"role": "user", "content": message})

    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, stream=True
    )
    reply = ""
    for chunk in stream:
        reply += chunk.choices[0].delta.content or ""
        yield reply
```

### Token budget and context window limits
As conversations grow, the messages array grows too. Every model has a maximum context window (measured in tokens). For long conversations you must either:
- Truncate old history (drop the oldest turns)
- Summarize prior context before appending new turns
- Use a sliding window over recent turns only

### Dynamic system prompts
The system prompt doesn't have to be static. It can be constructed programmatically based on runtime context — including previous messages:

```python
def build_system_prompt(history):
    base = "You are a helpful assistant."
    if len(history) > 5:
        base += " The user has been chatting a while; be more concise."
    return base
```

This pattern enables persona-shifting, upsell logic, or context-aware behavior changes mid-conversation.

### `gr.ChatInterface` — additional parameters
`gr.ChatInterface` exposes several useful configuration options:

```python
gr.ChatInterface(
    fn=chat,
    title="My Chatbot",
    description="Powered by GPT",
    examples=["Hello!", "What can you do?"],
    additional_inputs=[gr.Dropdown(...)],
).launch()
```

---

## Day 4 — Tool Calling
**File:** `week2/day4.ipynb`

### What this day covers
Day 4 introduces one of the most important patterns in LLM engineering: **tool calling** (also called function calling). This lets the LLM decide at runtime that it needs to invoke an external function to answer a query, rather than generating text from its weights alone.

### The core idea
The LLM doesn't execute code. Instead, when it decides a tool is needed, it returns a structured JSON response describing which function to call and with what arguments. Your code then calls the actual function and sends the result back to the LLM, which incorporates it into its final reply.

```
User message
    ↓
LLM call (with tool definitions)
    ↓
LLM returns tool_call (not text)
    ↓
Your code executes the function
    ↓
Return tool result to LLM
    ↓
LLM generates final text response
```

### Step 1 — Write the actual Python function
```python
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400"}

def get_ticket_price(destination_city):
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown destination")
```

### Step 2 — Define the tool schema (JSON)
The schema tells the LLM what the function does, when to use it, and what parameters it takes. Descriptions are critical — the LLM reads them to decide whether to call the tool:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_price",
            "description": "Get the price of a return ticket to a destination city. Call this when the user asks about ticket prices or wants to know how much a flight costs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_city": {
                        "type": "string",
                        "description": "The city the customer wants to travel to"
                    }
                },
                "required": ["destination_city"]
            }
        }
    }
]
```

### Step 3 — Pass tools to the API call
```python
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools
)
```

### Step 4 — Detect and handle a tool call
The response has `finish_reason == "tool_calls"` when the LLM wants to invoke a function:

```python
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get("destination_city")
    price = get_ticket_price(city)
    return {
        "role": "tool",
        "content": json.dumps({"destination_city": city, "price": price}),
        "tool_call_id": tool_call.id
    }
```

### Step 5 — The two-phase `chat()` function
```python
def chat(message, history):
    messages = build_messages(history, message)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools
    )

    # Phase 1: check if LLM wants to use a tool
    if response.choices[0].finish_reason == "tool_calls":
        message_obj = response.choices[0].message
        tool_response = handle_tool_call(message_obj)
        # Add the LLM's tool_calls message and the tool result to history
        messages.append(message_obj)
        messages.append(tool_response)
        # Phase 2: second call to get the final text response
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )

    return response.choices[0].message.content
```

### Why tool calling matters
- Gives the LLM access to real-time or private data it wasn't trained on
- Enables actions: booking, database writes, API calls
- Foundation for agentic systems where the LLM decides what steps to take
- The LLM remains in charge of *when* to call the tool and *how* to incorporate the result

### Gradio UI for the Day 4 chatbot
The airline ticket price chatbot is wrapped in a `gr.ChatInterface`, so the user can ask natural-language questions ("How much is a flight to Tokyo?") and the LLM transparently invokes `get_ticket_price` behind the scenes:

```python
gr.ChatInterface(fn=chat, title="Airline AI Assistant").launch()
```

---

## Day 5 — Multi-Modal Assistant: Images and Audio
**File:** `week2/day5.ipynb`

### What this day covers
Day 5 extends the Day 4 airline chatbot into a fully multi-modal application, adding image generation (DALL-E) and text-to-speech (OpenAI TTS), and building a more complex Gradio UI using `gr.Blocks` to display images alongside the chat.

### Adding a second tool — `book_ticket`
The assistant gains a booking tool alongside the price lookup tool, demonstrating multiple tools in one `tools` list:

```python
def book_ticket(destination_city):
    # simulates making a booking
    return f"Booking confirmed for {destination_city}!"

# tools list now contains both get_ticket_price and book_ticket schemas
```

When multiple tools are defined, `handle_tool_call` must dispatch to the right function:
```python
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if name == "get_ticket_price":
        result = get_ticket_price(arguments["destination_city"])
    elif name == "book_ticket":
        result = book_ticket(arguments["destination_city"])

    return {"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id}
```

### Image generation with DALL-E
The `artist()` function generates a travel image for the destination using OpenAI's image generation API:

```python
def artist(city):
    image_response = openai_client.images.generate(
        model="dall-e-3",
        prompt=f"A stunning travel photograph of {city}, photorealistic",
        size="1024x1024",
        n=1
    )
    image_url = image_response.data[0].url
    return image_url
```

The returned URL is passed to the Gradio UI for display alongside the chat.

### Text-to-speech with OpenAI TTS
The `talker()` function converts the assistant's text reply into spoken audio using the OpenAI TTS API:

```python
def talker(message):
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=message
    )
    audio_stream = BytesIO(response.content)
    audio_data, sample_rate = sf.read(audio_stream)
    return (sample_rate, audio_data)
```

The returned audio data is passed to a `gr.Audio` component in Gradio.

### `gr.Blocks` — composing custom layouts
When you need multiple output types (chat + image + audio), `gr.Blocks` replaces `gr.ChatInterface` and gives you full layout control:

```python
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500)
        image_output = gr.Image(height=500)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        msg = gr.Textbox()
    msg.submit(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, image_output, audio_output]
    )

ui.launch()
```

### Updated `chat()` function — returns multiple outputs
With a multi-output Gradio interface, `chat()` must return a tuple matching the outputs list:

```python
def chat(message, history):
    # ... build messages, handle tool calls ...
    reply = response.choices[0].message.content

    image = artist(city) if destination_mentioned else None
    audio = talker(reply)

    history.append([message, reply])
    return "", history, image, audio
```

### The complete Day 5 application stack
```
User types a message
    ↓
chat() builds messages + calls OpenAI with tools
    ↓
LLM returns tool_call (price or booking)
    ↓
handle_tool_call() executes the function, returns result
    ↓
Second LLM call generates the final reply text
    ↓
artist() generates a destination image (DALL-E)
talker() converts reply to speech (TTS)
    ↓
Gradio Blocks displays: chat history + image + audio
```

---

## Week 2 Concept Summary

### The progression across the week
| Day | Focus | New capability |
|---|---|---|
| Day 1 | Multi-provider API | All four providers side-by-side, streaming each |
| Day 2 | Gradio `ChatInterface` | History-aware chatbot UI with provider dropdown |
| Day 3 | Conversation management | History reconstruction, token limits, dynamic system prompts |
| Day 4 | Tool calling | Two-phase LLM call, JSON tool schema, `handle_tool_call()` |
| Day 5 | Multi-modal | DALL-E images, TTS audio, `gr.Blocks` layout |

### Tool calling sequence
```
messages + tools list
    ↓ first API call
finish_reason == "tool_calls"
    ↓ your code runs the function
tool result added to messages
    ↓ second API call
final text response
```

### Gradio component reference
| Component | Used in | Purpose |
|---|---|---|
| `gr.Interface` | Week 1 | Simple one-shot function → UI |
| `gr.ChatInterface` | Day 2, 3, 4 | Full chat UI with automatic history |
| `gr.Blocks` | Day 5 | Custom multi-panel layout |
| `gr.Chatbot` | Day 5 | Chat history display inside Blocks |
| `gr.Dropdown` | Day 2 | Provider/model selector |
| `gr.Image` | Day 5 | Display generated images |
| `gr.Audio` | Day 5 | Play TTS audio output |
| `gr.Textbox` | Day 2+ | Message input field |

### Key terms introduced in Week 2
| Term | Definition |
|---|---|
| **Tool calling** | Pattern where the LLM requests execution of an external function |
| **Tool schema** | JSON definition of a tool's name, description, and parameters |
| **`finish_reason`** | Field on the API response indicating why generation stopped; `"tool_calls"` signals a function request |
| **Two-phase call** | First call returns a tool request; second call receives the result and generates the final reply |
| **DALL-E** | OpenAI's image generation model, accessible via `images.generate()` |
| **TTS** | Text-to-speech; OpenAI's `audio.speech.create()` converts text to spoken audio |
| **`gr.Blocks`** | Gradio's low-level layout API for building custom multi-component UIs |
| **Dynamic system prompt** | A system prompt built at runtime based on conversation state or context |
| **Context window** | Maximum tokens a model can process in one call; grows with conversation history |

---

*Notes compiled from `week2/day1.ipynb`, `week2/day2.ipynb`, `week2/day3.ipynb`, `week2/day4.ipynb`, and `week2/day5.ipynb` in the ed-donner/llm_engineering repo.*
