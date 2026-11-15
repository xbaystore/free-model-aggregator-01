# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/xbaystore/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/xbaystore/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, two APIs, ten free-tier providers. It serves the OpenAI Chat Completions API
(`POST /v1/chat/completions`) and the Anthropic Messages API (`POST /v1/messages`), each in JSON
and in server-sent events, routing every request to the best free model that can serve it, and
failing over when a provider refuses.

## Run it

```sh
node main.ts --list   # print the catalog the server found, then exit 0
node main.ts          # serve: reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL
```

Node 22, standard library only; nothing is installed. The server reads the trial's brief
(`$MAKEFUN_PROMPT_FILE`): the ten providers, the quality table and the workload manifest. It
probes each provider's catalog route in the provider's own wire, registers its port and catalog
with the env, serves the 20 requests, and exits 0 when the env says the workload is done.

Outside a trial, with no brief and no env set, `--list` prints `{"providers":[]}` and exits 0.

## How it routes

For each request, the candidates are the free models, across all ten providers, that meet the
request's requirements (tools, vision, JSON mode, minimum context), ordered by the quality
table's `rating_milli`, highest first. Ties go to the provider with the fewest calls made so far,
then by provider and model id. A provider's per-trial budget is never published, so the server
learns it only by being refused: a 402 or 429 moves the request to the next candidate, and a
second pass retries the refused candidates once, because a refusal can also be one call on a
provider's failure schedule rather than a spent budget. A provider that has refused three times is
treated as spent and sorts last in line (the trials draw at most two scheduled failures per
provider), so late requests stop burning calls on it; it stays a candidate in case every other
provider is spent too. A 400 or 401 retires that provider/model pair for the rest of the trial. At most one provider call ever answers a request.

## How it speaks the wires

The providers answer with one fixed text, `ANSWER ECHO{...}`, echoing what they parsed: the
system prompt, the user text, the tool names, `max_tokens`, the stop sequences, the image and
the JSON mode. The server carries every one of those fields across the wires unchanged, and
returns the provider's text whole with the provider's usage counts as that wire reports them:

- OpenAI, Anthropic and Gemini upstream wires, each with its provider's own auth header
  (`Authorization: Bearer`, `x-api-key`, `x-goog-api-key`, or none).
- A streamed OpenAI call carries `stream_options.include_usage`, or the provider sends no usage
  chunk and the usage check fails.
- JSON mode cannot ride the Anthropic wire, so a JSON-mode OpenAI request never routes to an
  anthropic-wire provider; Gemini carries it as `responseMimeType: application/json`.
- `GET /v1/models` answers in the wire its auth header names: `Authorization: Bearer` gets the
  OpenAI list, `x-api-key` the Anthropic one, each listing exactly the free models registered.

## Determinism

Every choice derives from the brief, the requests and the providers' answers: no wall clock, no
unseeded randomness. A replay of a trial sees the same provider answers and takes the same route
to the same state export.

## Files

- `main.ts` — the server: the brief, the catalog, the routes, the register and the poll.
- `lib/providers.ts` — the brief's types and parser, the auth headers, the catalog probe.
- `lib/wire.ts` — the client wires and the three upstream wires, and the translations between them.
- `LICENSE` — MIT.

## License

MIT. See `LICENSE`.
