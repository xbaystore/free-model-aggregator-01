// The client wires and the three upstream wires this server speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This server forwards each request to
// the routed provider on that provider's own wire and translates the answer back:
//
//   - openai client -> openai provider: the body passes through (model replaced), streamed or not;
//   - anthropic client -> openai provider: rewritten to Chat Completions and answered back;
//   - openai client -> anthropic provider: rewritten to Messages and answered back;
//   - either client -> gemini provider: rewritten to generateContent / streamGenerateContent;
//   - the gemini and anthropic answers are rewritten into the client's wire, JSON or SSE.
//
// The providers answer with one fixed text, `ANSWER ECHO{...}`, that echoes what they parsed from the
// request: the system prompt, the user text, the tool names, max_tokens, the stop sequences, the image
// and the JSON mode. The trial's client compares that echo with what it sent, so every translation below
// must carry those fields across the wires unchanged: the system text, the last user message's text, the
// tool names in order, max_tokens, the stops, the image bytes and the JSON mode. A field rewritten or
// dropped loses the request.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (dataUrl !== null) {
      return { type: 'image', source: { type: 'base64', media_type: dataUrl[1] ?? 'image/png', data: dataUrl[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as OpenAI's finish_reason. */
export function geminiFinishToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  if (finish === 'TOOL_CALLS') return 'tool_calls';
  return 'stop';
}

/** A Gemini finishReason as Anthropic's stop_reason. */
export function geminiFinishToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  if (finish === 'TOOL_CALLS') return 'tool_use';
  return 'end_turn';
}

/** The stop sequences of a request body, whichever wire spelled them. */
function stopsOf(body: Json): string[] {
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop_sequences)) return body.stop_sequences.filter((s): s is string => typeof s === 'string');
  return [];
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the system
 * messages become the system prompt, image_url parts become image blocks, tool messages become
 * tool_result blocks, tool_calls become tool_use blocks, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 *
 * JSON mode cannot ride the Anthropic wire (its parser never reports json_mode), so the caller must not
 * route a JSON-mode OpenAI request here; main.ts excludes anthropic-wire providers for those.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const content = raw.content;
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof content === 'string' ? content : toolResultText(content),
          },
        ],
      });
      continue;
    }
    // A user message: a string stays a string, a part array becomes Anthropic blocks.
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = blockOf(part);
        if (block !== null) blocks.push(block);
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini content part: text or inline image data; null for the rest. */
function geminiPartOfOpenAI(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (dataUrl !== null) return { inlineData: { mimeType: dataUrl[1] ?? 'image/png', data: dataUrl[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini content part: text or inline image data; null for the rest. */
function geminiPartOfAnthropic(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** The generationConfig a Gemini call carries: the output cap, sampling, stops and JSON mode. */
function geminiConfigOf(body: Json, jsonMode: boolean): Json {
  const config: Json = {};
  const max = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (max !== null) config.maxOutputTokens = max;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopsOf(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request. System messages become the
 * system instruction, user parts become user contents (text plus inline image data), assistant messages
 * become model contents, tool calls become functionCall parts, tools become functionDeclarations.
 */
export function openAIToGeminiRequest(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: '',
              response: { content: typeof content === 'string' ? content : toolResultText(content) },
            },
          },
        ],
      });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const geminiPart = geminiPartOfOpenAI(part);
        if (geminiPart !== null) parts.push(geminiPart);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const jsonMode = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const out: Json = { contents };
  const system = systemParts.join('\n');
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = Array.isArray(body.tools)
    ? body.tools.flatMap((raw) => {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
        return [
          {
            name: raw.function.name,
            description: typeof raw.function.description === 'string' ? raw.function.description : '',
            parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
          },
        ];
      })
    : [];
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config = geminiConfigOf(body, jsonMode);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request. The system prompt becomes the
 * system instruction, user blocks become user contents, tool_use blocks become functionCall parts,
 * tool_result blocks become functionResponse parts, tools become functionDeclarations.
 */
export function anthropicToGeminiRequest(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({
              functionCall: {
                name: typeof block.name === 'string' ? block.name : '',
                args: isRecord(block.input) ? block.input : {},
              },
            });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: '',
              response: { content: toolResultText(block.content) },
            },
          });
          continue;
        }
        const geminiPart = geminiPartOfAnthropic(block);
        if (geminiPart !== null) parts.push(geminiPart);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = Array.isArray(body.tools)
    ? body.tools.flatMap((raw) => {
        if (!isRecord(raw) || typeof raw.name !== 'string') return [];
        return [
          {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        ];
      })
    : [];
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config = geminiConfigOf(body, false);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a Gemini candidate's parts, joined as the provider wrote them. */
function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini candidate's finishReason, or null when the answer names none. */
function geminiFinishOf(body: Json): string | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  return typeof candidate.finishReason === 'string' ? candidate.finishReason : null;
}

/** A Gemini answer's usage as {input, output} token counts. */
function geminiUsageOf(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object naming `model`. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: geminiTextOf(body) },
        finish_reason: geminiFinishToOpenAI(geminiFinishOf(body)),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object naming `model`. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  const text = geminiTextOf(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiFinishToAnthropic(geminiFinishOf(body)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object naming `model`. */
export function anthropicToOpenAIAnswer(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: {
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: finishReasonOf(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One frame of an upstream event stream: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

function parseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/** Split an upstream event stream into frames, tolerating CRLF line endings. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield parseFrame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

/** One OpenAI chunk object: a delta, a finish_reason, or a usage-only chunk. */
function openAIChunk(model: string, delta: Json, finish: string | null, usage: Json | null): Json {
  const choice: Json = { index: 0, delta, finish_reason: finish };
  const chunk: Json = { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [choice] };
  if (usage !== null) chunk.usage = usage;
  return chunk;
}

/**
 * An Anthropic SSE stream as OpenAI Chat Completions chunks: a role chunk, content deltas (and tool-call
 * deltas where the provider sent any), a finish chunk, the usage chunk and [DONE]. `model` is the model
 * the client asked for, named back to it.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  let started = false;
  let finished = false;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();

  const roleChunk = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null, null))];
  };
  const finishChunk = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      sseEvent(null, openAIChunk(model, {}, finishReasonOf(stopReason), null)),
      sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } }),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const out: string[] = [];
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
        break;
      }
      case 'content_block_start': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use') {
          out.push(...roleChunk());
          const toolIndex = nextIndex;
          nextIndex += 1;
          toolIndexes.set(index, toolIndex);
          out.push(
            sseEvent(
              null,
              openAIChunk(model, { tool_calls: [{ index: toolIndex, id: typeof block.id === 'string' ? block.id : `toolu_${toolIndex}`, type: 'function', function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' } }] }, null, null),
            ),
          );
        }
        break;
      }
      case 'content_block_delta': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const delta = isRecord(data.delta) ? data.delta : {};
        out.push(...roleChunk());
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          if (textIndex === -1) {
            textIndex = nextIndex;
            nextIndex += 1;
          }
          out.push(sseEvent(null, openAIChunk(model, { content: delta.text }, null, null)));
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
          const toolIndex = toolIndexes.get(index);
          if (toolIndex !== undefined) {
            out.push(sseEvent(null, openAIChunk(model, { tool_calls: [{ index: toolIndex, function: { arguments: delta.partial_json } }] }, null, null)));
          }
        }
        break;
      }
      case 'message_delta': {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
        if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
        break;
      }
      case 'message_stop': {
        out.push(...roleChunk(), ...finishChunk());
        break;
      }
      default:
        break;
    }
    for (const event of out) yield event;
  }
  if (!finished) {
    for (const event of roleChunk()) yield event;
    for (const event of finishChunk()) yield event;
  }
}

/**
 * A Gemini SSE stream as OpenAI Chat Completions chunks: a role chunk, content deltas from the
 * candidates' parts, a finish chunk from the final frame's finishReason, the usage chunk from its
 * usageMetadata, and [DONE].
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let finish: string | null = null;
  let started = false;
  let finished = false;

  const roleChunk = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null, null))];
  };
  const finishChunk = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      sseEvent(null, openAIChunk(model, {}, finish ?? 'stop', null)),
      sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } }),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const out: string[] = [];
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const text = geminiTextOf(data);
    if (text !== '') {
      out.push(...roleChunk());
      out.push(sseEvent(null, openAIChunk(model, { content: text }, null, null)));
    }
    const frameFinish = geminiFinishOf(data);
    if (frameFinish !== null) finish = geminiFinishToOpenAI(frameFinish);
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
    if (frameFinish !== null) out.push(...finishChunk());
    for (const event of out) yield event;
  }
  if (!finished) {
    for (const event of roleChunk()) yield event;
    for (const event of finishChunk()) yield event;
  }
}

/**
 * A Gemini SSE stream as the Anthropic event sequence: message_start, content_block_start, text deltas,
 * content_block_stop, message_delta with the stop reason and the output tokens, then message_stop. The
 * input tokens arrive only with the final frame, so message_start carries 0; the trial's stream check
 * compares output tokens only.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let outputTokens = 0;
  let stopReason = 'end_turn';
  let started = false;
  let blockOpen = false;
  let finished = false;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      }),
      sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (blockOpen) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      blockOpen = false;
    }
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  for await (const frame of sseFrames(upstream)) {
    const out: string[] = [];
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const text = geminiTextOf(data);
    if (text !== '') {
      out.push(...startEvents());
      blockOpen = true;
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }));
    }
    const frameFinish = geminiFinishOf(data);
    if (frameFinish !== null) stopReason = geminiFinishToAnthropic(frameFinish);
    if (isRecord(data.usageMetadata)) outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    if (frameFinish !== null) out.push(...startEvents(), ...finishEvents());
    for (const event of out) yield event;
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents()) yield event;
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}
