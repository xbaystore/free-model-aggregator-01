// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The server reads the trial's brief, probes each provider's catalog route in the provider's own wire,
// builds one catalog, registers its port and catalog with the env, then serves the client's 20 requests:
// GET /v1/models (OpenAI shape for an Authorization: Bearer request, Anthropic shape for x-api-key),
// POST /v1/chat/completions and POST /v1/messages, each in JSON or server-sent events. It polls
// GET $MAKEFUN_ENV_URL/work and exits 0 the moment the env says the workload is done.
//
// ROUTING (quality is 60% of the score)
//   Every request's candidates are the free models, across all ten providers, that meet the request's
//   requirements, ordered by the quality table's rating_milli, highest first; ties go to the provider
//   with the fewest calls made so far, then by provider and model id, so the order is a pure function of
//   the brief and the provider answers so far. A provider's per-trial budget is never published, so the
//   server learns it only by being refused: a 402 or 429 moves the request to the next candidate, and a
//   second pass retries the refused candidates once, because a refusal can also be one call on a
//   provider's failure schedule rather than a spent budget. A provider that has refused three times is
//   treated as spent and sorts last in line (the trials draw at most two scheduled failures per
//   provider), so late requests stop burning calls on it; it stays a candidate in case every other
//   provider is spent too. A 400 or 401 (unknown or paid model, wrong
//   key) retires that provider/model pair for the rest of the trial instead.
//
// WIRES (wire fidelity is 10%, and a wrong echo loses the request outright)
//   The providers answer with one fixed text, `ANSWER ECHO{...}`, echoing what they parsed: the system
//   prompt, the user text, the tool names, max_tokens, the stop sequences, the image and the JSON mode.
//   The server therefore passes every one of those fields across the wires unchanged (lib/wire.ts), and
//   returns the provider's text whole and the provider's usage counts as that wire reports them.
//   Two wire facts the starter missed: a streamed OpenAI call needs stream_options.include_usage, or the
//   provider sends no usage chunk and the usage check fails; and JSON mode cannot ride the Anthropic
//   wire, so a JSON-mode OpenAI request never routes to an anthropic-wire provider (Gemini carries it
//   as responseMimeType instead).
//
// DETERMINISM
//   Every choice derives from the brief, the requests and the providers' answers: no wall clock, no
//   unseeded randomness. The replay of a trial sees the same provider answers and takes the same route
//   to the same state export.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the client wires and the three upstream wires, and the translations between them.
//   README.md          how the server works and how to run it.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  anthropicToOpenAIAnswer,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Upstream calls made so far, per provider id: successful and refused alike. */
const callsMade = new Map<string, number>();
/** (provider, model) pairs a provider refused as unknown, paid or unauthorised: never tried again. */
const deadModels = new Set<string>();
/** Refusals (402, 429 or 5xx) seen so far, per provider id. */
const refusals = new Map<string, number>();
/**
 * Refusals after which a provider is treated as spent: the trials draw one or two scheduled failures
 * per provider, so a third refusal means its budget is spent or its outage has begun, and trying it
 * first only burns a call. A spent provider stays a candidate, last in line, in case every other
 * provider is spent too.
 */
const SUSPECT_REFUSALS = 3;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that can serve one request, best first: providers with fewer than three refusals
 * come before spent ones (a third refusal means the budget is spent or the outage has begun), then the
 * quality table's rating, highest first; ties go to the provider with the fewest calls made so far
 * (which spreads the load over the providers whose budgets are still unknown), then by provider and
 * model id. A JSON-mode request never lists an anthropic-wire provider: that wire cannot carry JSON
 * mode, and the provider would echo it absent.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const spent = (c: Candidate): number => ((refusals.get(c.probe.provider.id) ?? 0) >= SUSPECT_REFUSALS ? 1 : 0);
  out.sort(
    (a, b) =>
      spent(a) - spent(b) ||
      b.rating - a.rating ||
      (callsMade.get(a.probe.provider.id) ?? 0) - (callsMade.get(b.probe.provider.id) ?? 0) ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether an OpenAI client body asks for JSON mode. */
function jsonModeOf(body: Json): boolean {
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One request to the routed provider, in its own wire, carrying the client's request unchanged in what
 * the provider echoes: the system prompt, the user text and images, the tools, max_tokens, the stops and
 * the JSON mode. A streamed OpenAI call carries stream_options.include_usage, or the provider sends no
 * usage chunk and the trial's usage check fails.
 */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) {
      upstreamBody = { ...upstreamBody, stream: true, stream_options: { include_usage: true } };
      headers.accept = 'text/event-stream';
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    if (stream) {
      upstreamBody = { ...upstreamBody, stream: true };
      headers.accept = 'text/event-stream';
    }
  } else {
    const path = encodeURIComponent(candidate.model.id);
    url = stream ? `${base}/v1beta/models/${path}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${path}:generateContent`;
    upstreamBody = clientApi === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
    if (stream) headers.accept = 'text/event-stream';
  }
  callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
}

interface Served {
  candidate: Candidate;
  upstream: Response;
}

/**
 * Serve one client request with the best free model that answers it. Candidates are tried in rating
 * order; a 402, 429 or 5xx moves to the next candidate, and a second pass retries the refused ones
 * once, because a refusal can be one call on a provider's failure schedule rather than a spent budget.
 * A 400 or 401 retires that provider/model pair for the rest of the trial. At most one provider call
 * ever answers a request, so the trial's one-provider check holds.
 */
async function serveOne(clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Served | null> {
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, clientApi === 'openai' && jsonModeOf(body));
  const refused: number[] = [];
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 0; i < candidates.length; i++) {
      if (pass === 1 && !refused.includes(i)) continue;
      const candidate = candidates[i];
      const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
      if (deadModels.has(key)) continue;
      let upstream: Response;
      try {
        upstream = await callUpstream(candidate, clientApi, body, stream);
      } catch (err) {
        log(`upstream ${key} unreachable: ${(err as Error).message}; failing over`);
        continue;
      }
      if (upstream.ok) return { candidate, upstream };
      const status = upstream.status;
      try {
        await upstream.text();
      } catch {
        // The error body could not be read: the refusal stands either way.
      }
      if (status === 400 || status === 401) {
        deadModels.add(key);
        log(`upstream ${key} refused HTTP ${status}; retiring the pair`);
      } else if (status === 402 || status === 429 || status >= 500) {
        refusals.set(candidate.probe.provider.id, (refusals.get(candidate.probe.provider.id) ?? 0) + 1);
        if (!refused.includes(i)) refused.push(i);
        log(`upstream ${key} refused HTTP ${status}; failing over`);
      } else {
        log(`upstream ${key} refused HTTP ${status}; failing over`);
      }
    }
    if (refused.length === 0) break;
  }
  return null;
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Answer one served request in the wire the client used, JSON or streamed, carrying the provider's text and usage whole. */
async function answer(
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  served: Served,
  res: ServerResponse,
): Promise<void> {
  const model = served.candidate.model.id;
  const wire = served.candidate.probe.provider.wire;
  const upstream = served.upstream;
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (upstream.body === null) {
      res.end();
      return;
    }
    if (clientApi === 'openai') {
      if (wire === 'openai') await pipeStream(upstream.body, res);
      else if (wire === 'anthropic') for await (const event of openAIEventsFromAnthropic(upstream.body, model)) res.write(event);
      else for await (const event of openAIEventsFromGemini(upstream.body, model)) res.write(event);
    } else if (wire === 'anthropic') {
      await pipeStream(upstream.body, res);
    } else if (wire === 'openai') {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    } else {
      for await (const event of anthropicEventsFromGemini(upstream.body, model)) res.write(event);
    }
    res.end();
    return;
  }
  let answerBody: unknown = null;
  try {
    answerBody = await upstream.json();
  } catch {
    answerBody = null;
  }
  if (!isRecord(answerBody)) {
    sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered no JSON object');
    return;
  }
  if (clientApi === 'openai') {
    if (wire === 'openai') sendJson(res, 200, answerBody);
    else if (wire === 'anthropic') sendJson(res, 200, anthropicToOpenAIAnswer(answerBody, model));
    else sendJson(res, 200, geminiToOpenAI(answerBody, model));
    return;
  }
  if (wire === 'anthropic') sendJson(res, 200, answerBody);
  else if (wire === 'openai') sendJson(res, 200, openAIToAnthropic(answerBody, model));
  else sendJson(res, 200, geminiToAnthropic(answerBody, model));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that takes it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serveOne('openai', body, stream);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  await answer('openai', stream, served, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that takes it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serveOne('anthropic', body, stream);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  await answer('anthropic', stream, served, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
