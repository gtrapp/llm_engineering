# LLM Engineering — Week 1 Study Notes
**Course:** AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents (Ed Donner / Udemy)
**Repo:** https://github.com/ed-donner/llm_engineering/tree/main/week1

---

## Big Picture: What Week 1 Is About

Week 1 establishes the foundational skills every LLM engineer needs before tackling advanced topics. By the end of the week you should be able to call multiple LLM provider APIs, write effective prompts, scrape and clean web content for use as LLM input, stream responses, and build a real working application (a website summarizer). Each day builds directly on the last.

---

## Day 1 — Your First LLM Application: Web Summarizer

**File:** `week1/day1.ipynb`

### Core concepts

**What is an LLM API?**  
A Large Language Model API lets you send text (a "prompt") to a hosted AI model and receive generated text back. You communicate through structured HTTP calls — no ML knowledge required to use one.

**The OpenAI message format**  
All modern frontier APIs use a messages array with roles:
- `system` — sets the assistant's persona/task instructions (runs before user input)
- `user` — the actual prompt/question from the caller
- `assistant` — the model's prior responses (used in multi-turn conversations)

```python
messages = [
    {"role": "system", "content": "You are an expert summarizer..."},
    {"role": "user",   "content": "Please summarize this page: ..."}
]
```

**Environment & API key management**  
- Store secrets in a `.env` file: `OPENAI_API_KEY=sk-...`
- Load at runtime with `python-dotenv`: `load_dotenv()`
- Never hard-code keys in notebooks

**Web scraping with `requests` + `BeautifulSoup`**  
- `requests.get(url)` fetches raw HTML
- `BeautifulSoup(html, "html.parser")` parses it into a traversable tree
- Strip noise: remove `<script>`, `<style>`, `<img>` tags before passing to the LLM
- Extract title: `soup.title.string`
- Extract body text: `soup.body.get_text()`

**The `Website` class pattern**  
A clean OOP wrapper that encapsulates fetch + parse logic:
```python
class Website:
    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        self.title = soup.title.string
        for tag in soup(["script","style","img"]):
            tag.decompose()
        self.text = soup.body.get_text()
```

**Prompt engineering basics**  
- The system prompt defines the assistant's role and output format
- The user prompt injects the scraped content
- Keep prompts concise — LLMs charge by token and have context limits

**Calling the OpenAI API**
```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
summary = response.choices[0].message.content
```

**Displaying output in Jupyter**  
Use `IPython.display.Markdown()` to render markdown formatting in notebook output.

### Key patterns introduced
- Extract → Process → Prompt → Display pipeline
- Separation of concerns: scraping class vs. prompt construction vs. API call vs. display

### Things to experiment with
- Change the `system_prompt` to alter tone (formal, bullet points, ELI5)
- Try different URLs and observe how HTML noise affects output quality
- Try `gpt-4o` vs `gpt-4o-mini` and compare quality vs. cost

---

## Day 2 — Alternative Models & Ollama (Local LLMs)

**File:** `week1/day2 EXERCISE.ipynb`

### Core concepts

**Why use multiple providers?**  
Different providers offer different trade-offs in quality, cost, speed, and data privacy. An LLM engineer needs to be fluent across all of them.

**The OpenAI-compatible API pattern**  
Many providers (Ollama, DeepSeek, Google via their OpenAI-compatible endpoint) accept requests in the same format as OpenAI. You only need to change `base_url` and `api_key`:

```python
# Ollama (local)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # placeholder — Ollama doesn't require a real key
)

# DeepSeek
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"]
)
```

**Ollama — running open-source models locally**  
- Ollama is a tool that lets you run models like `llama3.2` on your own machine
- Privacy advantage: data never leaves your computer
- Cost advantage: free to run (GPU/CPU compute only)
- Tradeoff: slower, requires sufficient local hardware
- Start the Ollama server, then call it exactly like OpenAI

**Model selection**  
| Provider | Model name | Notes |
|---|---|---|
| OpenAI | `gpt-4o-mini` | Fast, cheap, great for most tasks |
| OpenAI | `gpt-4o` | More capable, costs more |
| Anthropic | `claude-3-haiku-20240307` | Fast/cheap Claude option |
| Google | `gemini-2.0-flash` | Google's fast model |
| Ollama | `llama3.2` | Local, free, private |
| Ollama | `llama3.2:1b` | Tiny model, very fast locally |

**Exercise goal**  
Modify the Day 1 website summarizer to use Ollama/llama3.2 instead of OpenAI. This reinforces that your application logic is model-agnostic.

### Key insight
The OpenAI Python SDK is becoming the *de facto* standard interface. Knowing how to point it at different providers is a core LLM engineering skill.

---

## Day 3 — Anthropic Claude API & Multi-Provider Patterns

**File:** `week1/day3.ipynb` (content also covered in day1 and day2 EXERCISE)

### Core concepts

**The Anthropic API — key differences from OpenAI**  
Anthropic's `claude` client has a slightly different structure:
- `system` is a top-level parameter, not a message in the array
- Messages array only contains `user` and `assistant` turns
- Response accessed via `response.content[0].text` (not `.choices[0].message.content`)

```python
import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    system="You are a helpful assistant...",
    messages=[{"role": "user", "content": "Summarize this..."}]
)
summary = response.content[0].text
```

**Comparing providers side-by-side**  
A key exercise is calling the same prompt against multiple providers and comparing outputs — quality, style, length, and accuracy can differ meaningfully.

**API key management for multiple providers**  
```
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
DEEPSEEK_API_KEY=...
```

**Google Gemini — two integration options**  
1. Native Google SDK: `import google.generativeai as genai`
2. Via OpenAI client with Google's OpenAI-compatible base URL (simpler for comparison)

**Prompt engineering — system prompt design**  
- Be explicit about the format you want (markdown, JSON, bullet points)
- Specify length constraints ("in 3 sentences", "under 200 words")
- Assign a persona ("You are a senior engineer reviewing...")
- Anthropic's prompt generator tool can help craft better system prompts

### Key insight
Each provider has quirks in their API structure. Building a lightweight abstraction layer (a function that normalizes calls to any provider) is a pattern used heavily throughout this course.

---

## Day 4 — Streaming Responses & the Company Brochure Project

**File:** `week1/day4.ipynb`

### Core concepts

**What is streaming and why does it matter?**  
By default, the API waits until the full response is generated before returning it. With streaming, tokens are returned incrementally as they're generated — like watching text appear word-by-word. This dramatically improves perceived responsiveness in real applications.

**OpenAI streaming**
```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

**Anthropic streaming**  
Uses a context manager pattern with `.stream()`:
```python
with client.messages.stream(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    messages=messages
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

**The Company Brochure project**  
This is the day's capstone: build a tool that takes a company's homepage URL and generates a marketing brochure summary. It extends the Day 1 summarizer with:
- Multiple page scraping (follow relevant internal links)
- A more sophisticated system prompt targeting brochure-style output
- Streaming output for real-time display (`stream_brochure()` function)

**Link extraction pattern**  
```python
links = [a['href'] for a in soup.find_all('a', href=True)]
# Filter to internal links, scrape each, combine text
```

**Token limits & context windows**  
- Every model has a maximum context window (e.g., 128k tokens for GPT-4o)
- 1 token ≈ 0.75 words in English
- You must truncate or summarize source content before it exceeds the limit
- Scraping multiple pages requires careful text management

### Key patterns introduced
- Streaming response processing
- Multi-page content aggregation
- Extended prompt design for long-form output

---

## Day 5 — Gradio UI & Building a Polished Application

**File:** `week1/day5.ipynb`

### Core concepts

**Gradio — instant UIs for LLM apps**  
Gradio is a Python library that turns a Python function into a working web UI in ~1 line of code. No HTML/CSS/JS required.

```python
import gradio as gr

def summarize_website(url):
    # ... your summarization logic ...
    return summary

gr.Interface(
    fn=summarize_website,
    inputs=gr.Textbox(label="URL"),
    outputs=gr.Markdown(label="Summary")
).launch()
```

This creates a local web server with a form and output display — shareable via a public Gradio link.

**Two Gradio patterns used in the course**

1. `gr.Interface` — simple function wrapper, one input → one output
2. `gr.ChatInterface` — full chat UI with history management built in (covered more in Week 2)

**Streaming in Gradio**  
Gradio supports streaming output by using Python generators (`yield` instead of `return`):
```python
def stream_summary(url):
    stream = client.chat.completions.create(..., stream=True)
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result  # update the UI incrementally
```

**Multi-provider Gradio app**  
The Day 5 notebook brings together everything from the week: a Gradio app that lets the user pick a provider (OpenAI, Claude, Ollama) and model from a dropdown, then streams the summary. The `stream_gpt()` and `stream_claude()` functions are the core connectors.

**The `chat` function pattern (preview of Week 2)**  
```python
def chat(message, history, provider):
    # build messages from history
    # delegate to stream_gpt or stream_claude
    # yield incremental results
```

**Week 1 Exercise / Homework**  
The `week1 EXERCISE.ipynb` file is the homework: build something novel with what you've learned. Suggested extensions: add more providers, support PDF input, add language translation, export to file, add UI polish.

### Key patterns introduced
- Gradio `Interface` and `ChatInterface`
- Generator-based streaming in Gradio
- Multi-provider UI with dropdown selection
- End-to-end application combining all week's skills

---

## Week 1 Core Concepts Summary

### The LLM API Request/Response Cycle
```
You → messages[] → API → model inference → response → you
```
Every frontier API follows this pattern. The differences are in SDK syntax, authentication, and response object structure.

### Prompt Anatomy
| Part | Role | Tips |
|---|---|---|
| System prompt | Define persona, constraints, format | Be specific; set the output format here |
| User prompt | Inject content + task | Include all context the model needs |
| Few-shot examples | Show the model what you want | Optional but powerful for consistent output |

### Provider Quick-Reference
| Provider | SDK | System prompt location | Response access |
|---|---|---|---|
| OpenAI | `openai.OpenAI()` | In messages array, role="system" | `response.choices[0].message.content` |
| Anthropic | `anthropic.Anthropic()` | Top-level `system=` param | `response.content[0].text` |
| Google | `genai` or OpenAI compat | `system_instruction=` param | varies |
| Ollama | OpenAI client, local URL | Same as OpenAI | Same as OpenAI |

### The Week 1 Application Stack
```
Web URL
  ↓ requests.get()
Raw HTML
  ↓ BeautifulSoup (parse + clean)
Clean text
  ↓ prompt construction (system + user)
Messages array
  ↓ LLM API call (OpenAI / Claude / Ollama)
Generated summary
  ↓ Gradio UI / IPython.display
User
```

---

## Key Terms Glossary

| Term | Definition |
|---|---|
| **Token** | Smallest unit of text a model processes (~0.75 words) |
| **Context window** | Maximum tokens a model can process in one call |
| **System prompt** | Instructions that define the assistant's behavior |
| **Temperature** | Controls randomness (0 = deterministic, 1 = creative) |
| **Streaming** | Returning tokens incrementally as they're generated |
| **Ollama** | Tool for running open-source LLMs locally |
| **BeautifulSoup** | Python library for parsing HTML |
| **Gradio** | Python library for instant ML/LLM web UIs |
| **`.env` file** | File for storing API keys and secrets locally |
| **OpenAI-compatible API** | Any API that accepts the OpenAI message format |

---

## Recommended Study Path

1. Run `day1.ipynb` top to bottom, reading every cell carefully
2. Modify the system prompt — observe how output changes
3. Try `day2 EXERCISE.ipynb` — get Ollama running locally
4. Run through `day5.ipynb` to see the Gradio UI come together
5. Attempt the Week 1 Exercise — build something of your own
6. Review the community contributions folder for creative extensions

---

*Notes compiled from the ed-donner/llm_engineering repo, DeepWiki analysis, and edwarddonner.com course resources.*
