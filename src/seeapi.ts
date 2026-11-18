import type * as http from "node:http";
import crypto from "node:crypto";
import type {
  AsyncImageResponse,
  ChatCompletionRequest,
  Fixture,
  FixtureResponse,
  HandlerDefaults,
  ImageItem,
  RawJSONResponse,
  SeeapiModelInfo,
} from "./types.js";
import {
  flattenHeaders,
  getContext,
  getTestId,
  isErrorResponse,
  isImageResponse,
  isJSONResponse,
  isJsonObject,
  resolveResponse,
  strictNoMatchLogLine,
  strictNoMatchMessage,
} from "./helpers.js";
import { renderError } from "./response/errors.js";
import { resolveRequestDecisions, strictOverrideField } from "./response/policy.js";
import { matchFixtureDiagnostic } from "./router.js";
import { calculateDelay, delay, writeErrorResponse } from "./sse-writer.js";
import type { Journal } from "./journal.js";
import { applyChaosAsync, responseGone } from "./chaos.js";
import { resolveProgression } from "./fal.js";

/**
 * SeeAPI async image generation mock.
 *
 * - Submit: `POST /seeapi/v1/generations` (requires `Idempotency-Key` header,
 *   JSON body, strict input-field validation by default) -> HTTP 202
 *   `{ generation: { id, status: "queued", ... } }`.
 * - Poll: `GET /seeapi/v1/generations/{id}` -> HTTP 200
 *   `{ generation: { id, status, result?: { assets: [{ url }] }, error_message? } }`.
 * - Models: `GET /seeapi/v1/models` -> static configurable model list.
 * - List endpoint `GET /seeapi/v1/generations` is NOT served (404).
 */

export const SEEAPI_MAX_ENTRIES = 10_000;
const SEEAPI_TTL_MS = 3_600_000; // 1 hour

export const DEFAULT_SEEAPI_INPUT_FIELDS: readonly string[] = [
  "prompt",
  "input_images",
  "aspect_ratio",
  "resolution",
  "output_format",
];

export const DEFAULT_SEEAPI_MODELS: readonly SeeapiModelInfo[] = [
  {
    id: "nano-banana-2-1",
    provider: "seeapi",
    endpoints: ["text-to-image", "image-to-image"],
  },
];

const SEEAPI_DEFAULT_IMAGE_URL = "https://assets.seeapi.com/aimock-placeholder-image.png";

export type SeeapiTerminalFailureStatus =
  | "failed"
  | "canceled"
  | "cancelled"
  | "refunded"
  | "expired";

export type SeeapiStatus = "queued" | "processing" | "succeeded" | SeeapiTerminalFailureStatus;

export interface SeeapiSubmitReply {
  id: string;
  status: "queued" | "processing";
  consumed_credits?: number;
}

export interface SeeapiJob {
  id: string;
  model: string;
  status: SeeapiStatus;
  terminalStatus: "succeeded" | SeeapiTerminalFailureStatus;
  urls: string[];
  errorMessage?: string;
  consumedCredits?: number;
  pollCount: number;
  pollsBeforeInProgress: number;
  pollsBeforeCompleted: number;
  fixture: Fixture | null;
  submitReply: SeeapiSubmitReply;
}

interface SeeapiEntry {
  job: SeeapiJob;
  createdAt: number;
}

/**
 * Bounded per-testId job and idempotency store for SeeAPI generations.
 * Mirrors GrokVideoJobMap: lazy TTL eviction on `get`, FIFO eviction when
 * over capacity, world-generation counter for reset-mid-flight safety.
 */
export class SeeapiJobMap {
  private readonly entries = new Map<string, SeeapiEntry>();
  private readonly idempotencyKeys = new Map<string, { id: string; createdAt: number }>();
  private worldGeneration = 0;

  get generation(): number {
    return this.worldGeneration;
  }

  get(key: string): SeeapiJob | undefined {
    const entry = this.entries.get(key);
    if (!entry) return undefined;
    if (Date.now() - entry.createdAt > SEEAPI_TTL_MS) {
      this.entries.delete(key);
      return undefined;
    }
    return entry.job;
  }

  set(key: string, job: SeeapiJob): void {
    this.entries.delete(key);
    this.entries.set(key, { job, createdAt: Date.now() });
    if (this.entries.size > SEEAPI_MAX_ENTRIES) {
      const excess = this.entries.size - SEEAPI_MAX_ENTRIES;
      const iter = this.entries.keys();
      for (let i = 0; i < excess; i++) {
        const next = iter.next();
        if (!next.done) this.entries.delete(next.value);
      }
    }
  }

  getIdempotency(key: string): string | undefined {
    const entry = this.idempotencyKeys.get(key);
    if (!entry) return undefined;
    if (Date.now() - entry.createdAt > SEEAPI_TTL_MS) {
      this.idempotencyKeys.delete(key);
      return undefined;
    }
    return entry.id;
  }

  setIdempotency(key: string, id: string): void {
    this.idempotencyKeys.delete(key);
    this.idempotencyKeys.set(key, { id, createdAt: Date.now() });
    if (this.idempotencyKeys.size > SEEAPI_MAX_ENTRIES) {
      const excess = this.idempotencyKeys.size - SEEAPI_MAX_ENTRIES;
      const iter = this.idempotencyKeys.keys();
      for (let i = 0; i < excess; i++) {
        const next = iter.next();
        if (!next.done) this.idempotencyKeys.delete(next.value);
      }
    }
  }

  delete(key: string): boolean {
    return this.entries.delete(key);
  }

  clear(): void {
    this.entries.clear();
    this.idempotencyKeys.clear();
    this.worldGeneration++;
  }

  get size(): number {
    return this.entries.size;
  }
}

// ─── Shared async-image fixture helpers (SeeAPI + KIE) ───────────────────────

export interface ExtractedAsyncImageOutcome {
  urls: string[];
  error?: string;
  failCode?: string;
  status?: SeeapiStatus;
  consumedCredits?: number;
}

function urlsFromImageItems(items: ImageItem[] | undefined): string[] {
  if (!Array.isArray(items)) return [];
  return items
    .map((item) => item.url)
    .filter((u): u is string => typeof u === "string" && u.length > 0);
}

/**
 * Normalize an `AsyncImageResponse` into a `RawJSONResponse` so it passes
 * fixture validation (`validateFixtures`) and flows cleanly through `resolveResponse`.
 */
export function asyncImageResponseToJson(response: AsyncImageResponse): RawJSONResponse {
  const urls = Array.isArray(response.urls)
    ? response.urls
    : response.images
      ? urlsFromImageItems(response.images)
      : response.image?.url
        ? [response.image.url]
        : [];
  const payload: Record<string, unknown> = { urls };
  if (response.error !== undefined) payload.error = response.error;
  if (response.failCode !== undefined) payload.failCode = response.failCode;
  if (response.status !== undefined) payload.status = response.status;
  if (response.consumedCredits !== undefined) payload.consumedCredits = response.consumedCredits;
  return { json: payload };
}

/**
 * Extract image URLs and optional failure metadata from a resolved fixture
 * response (`ImageResponse` or `RawJSONResponse`). Returns `null` when the
 * response type is unsupported.
 */
export function extractAsyncImageOutcome(
  response: FixtureResponse,
  defaultUrl: string,
): ExtractedAsyncImageOutcome | null {
  if (isImageResponse(response)) {
    const items = response.images ?? (response.image ? [response.image] : []);
    const urls = urlsFromImageItems(items);
    return { urls: urls.length > 0 ? urls : [defaultUrl] };
  }
  if (!isJSONResponse(response)) return null;

  const raw = response.json;
  if (!isJsonObject(raw)) return null;

  const urls: string[] = [];
  if (Array.isArray(raw.urls)) {
    for (const u of raw.urls) {
      if (typeof u === "string" && u.length > 0) urls.push(u);
    }
  } else if (Array.isArray(raw.resultUrls)) {
    for (const u of raw.resultUrls) {
      if (typeof u === "string" && u.length > 0) urls.push(u);
    }
  } else if (Array.isArray(raw.images)) {
    for (const item of raw.images) {
      if (isJsonObject(item) && typeof item.url === "string" && item.url.length > 0) {
        urls.push(item.url);
      }
    }
  }

  const error = typeof raw.error === "string" ? raw.error : undefined;
  const failCode = typeof raw.failCode === "string" ? raw.failCode : undefined;
  const status = typeof raw.status === "string" ? (raw.status as SeeapiStatus) : undefined;
  const consumedCredits =
    typeof raw.consumedCredits === "number" && Number.isFinite(raw.consumedCredits)
      ? raw.consumedCredits
      : undefined;

  const isFailure =
    error !== undefined ||
    (status !== undefined &&
      status !== "succeeded" &&
      status !== "queued" &&
      status !== "processing");

  const outcome: ExtractedAsyncImageOutcome = {
    urls: isFailure || urls.length > 0 ? urls : [defaultUrl],
  };
  if (error !== undefined) outcome.error = error;
  if (failCode !== undefined) outcome.failCode = failCode;
  if (status !== undefined) outcome.status = status;
  if (consumedCredits !== undefined) outcome.consumedCredits = consumedCredits;
  return outcome;
}

// ─── SeeAPI lifecycle helpers ────────────────────────────────────────────────

const SEEAPI_FAILURE_STATUSES = new Set<string>([
  "failed",
  "canceled",
  "cancelled",
  "refunded",
  "expired",
]);

function isSeeapiTerminal(status: SeeapiStatus): boolean {
  return status === "succeeded" || SEEAPI_FAILURE_STATUSES.has(status);
}

function advanceSeeapiJob(job: SeeapiJob): void {
  if (isSeeapiTerminal(job.status)) return;
  job.pollCount += 1;
  if (job.status === "queued" && job.pollCount >= job.pollsBeforeInProgress) {
    job.status = "processing";
  } else if (job.pollCount >= job.pollsBeforeCompleted) {
    job.status = job.terminalStatus;
  }
}

function serializeSeeapiPoll(job: SeeapiJob): Record<string, unknown> {
  const generation: Record<string, unknown> = {
    id: job.id,
    status: job.status,
  };
  if (job.consumedCredits !== undefined) {
    generation.consumed_credits = job.consumedCredits;
  }
  if (job.status === "succeeded") {
    generation.result = {
      assets: job.urls.map((url) => ({ url })),
    };
  } else if (SEEAPI_FAILURE_STATUSES.has(job.status)) {
    generation.error_message = job.errorMessage || "Generation failed";
  }
  return { generation };
}

function headerSingle(headers: http.IncomingHttpHeaders, name: string): string | undefined {
  const val = headers[name];
  if (Array.isArray(val)) return val[0];
  return val;
}

function validationJournalBody(rawObj: Record<string, unknown>): ChatCompletionRequest {
  const rawModel = rawObj.model;
  const model =
    typeof rawModel === "string"
      ? rawModel
      : rawModel === undefined
        ? ""
        : JSON.stringify(rawModel);
  const sanitized = Object.fromEntries(
    Object.entries(rawObj).filter(([key]) => !key.startsWith("_")),
  );
  return { ...sanitized, model, messages: [] };
}

function sendSeeapiValidationError(
  res: http.ServerResponse,
  status: number,
  message: string,
  code: string,
  journal: Journal,
  journalMeta: {
    method: string;
    path: string;
    headers: Record<string, string>;
    body: ChatCompletionRequest | null;
  },
): void {
  journal.add({
    ...journalMeta,
    service: "seeapi",
    response: { status, fixture: null },
  });
  writeErrorResponse(
    res,
    status,
    renderError("openai", "validation", {
      message,
      type: "invalid_request_error",
      code,
    }).body,
  );
}

// ─── POST /seeapi/v1/generations — submit ────────────────────────────────────

export async function handleSeeapiCreate(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  raw: string,
  fixtures: Fixture[],
  journal: Journal,
  defaults: HandlerDefaults,
  setCorsHeaders: (res: http.ServerResponse) => void,
  jobs: SeeapiJobMap,
): Promise<void> {
  setCorsHeaders(res);
  const path = req.url ?? "/seeapi/v1/generations";
  const method = req.method ?? "POST";
  const headers = flattenHeaders(req.headers);

  if (defaults.seeapi?.requireAuth) {
    const auth = headerSingle(req.headers, "authorization")?.trim() ?? "";
    const bearer = /^Bearer\s+(\S.*)$/i.exec(auth);
    if (!bearer) {
      journal.add({
        method,
        path,
        headers,
        body: null,
        service: "seeapi",
        response: { status: 401, fixture: null },
      });
      writeErrorResponse(
        res,
        401,
        renderError("openai", "authentication", {
          message: "Missing or invalid Authorization header",
        }).body,
      );
      return;
    }
  }

  const contentType = headerSingle(req.headers, "content-type") ?? "";
  if (contentType.toLowerCase().includes("multipart/form-data")) {
    sendSeeapiValidationError(
      res,
      400,
      "multipart/form-data is not supported — SeeAPI is JSON-only",
      "invalid_request",
      journal,
      { method, path, headers, body: null },
    );
    return;
  }

  const rawIdempotencyKey = headerSingle(req.headers, "idempotency-key");
  const idempotencyKey = typeof rawIdempotencyKey === "string" ? rawIdempotencyKey.trim() : "";
  if (!idempotencyKey) {
    sendSeeapiValidationError(
      res,
      400,
      "Missing required header: Idempotency-Key",
      "missing_idempotency_key",
      journal,
      { method, path, headers, body: null },
    );
    return;
  }

  let parsedRaw: unknown;
  try {
    parsedRaw = JSON.parse(raw);
  } catch (parseErr) {
    const detail = parseErr instanceof Error ? parseErr.message : "unknown";
    sendSeeapiValidationError(res, 400, `Malformed JSON: ${detail}`, "invalid_json", journal, {
      method,
      path,
      headers,
      body: null,
    });
    return;
  }

  if (!isJsonObject(parsedRaw)) {
    sendSeeapiValidationError(
      res,
      400,
      "Request body must be a JSON object",
      "invalid_request",
      journal,
      { method, path, headers, body: null },
    );
    return;
  }

  const journalBody = validationJournalBody(parsedRaw);

  if (typeof parsedRaw.model !== "string" || !parsedRaw.model.trim()) {
    sendSeeapiValidationError(
      res,
      400,
      "Missing or invalid parameter: 'model' must be a non-empty string",
      "invalid_request",
      journal,
      { method, path, headers, body: journalBody },
    );
    return;
  }
  const model = parsedRaw.model.trim();

  if (!isJsonObject(parsedRaw.input)) {
    sendSeeapiValidationError(
      res,
      400,
      "Missing or invalid parameter: 'input' must be a JSON object",
      "invalid_request",
      journal,
      { method, path, headers, body: journalBody },
    );
    return;
  }
  const input = parsedRaw.input;

  if (defaults.seeapi?.strictInputFields !== false) {
    const allowedList =
      defaults.seeapi?.allowedInputFieldsByModel?.[model] ??
      defaults.seeapi?.defaultAllowedInputFields ??
      DEFAULT_SEEAPI_INPUT_FIELDS;
    const allowed = new Set(allowedList);
    for (const key of Object.keys(input)) {
      if (!allowed.has(key)) {
        sendSeeapiValidationError(
          res,
          400,
          `invalid input: unknown input field ${key}`,
          "invalid_input",
          journal,
          { method, path, headers, body: journalBody },
        );
        return;
      }
    }
  }

  if (input.prompt !== undefined && typeof input.prompt !== "string") {
    sendSeeapiValidationError(
      res,
      400,
      "Invalid type for parameter: 'input.prompt' must be a string",
      "invalid_input",
      journal,
      { method, path, headers, body: journalBody },
    );
    return;
  }

  if (input.input_images !== undefined) {
    if (
      !Array.isArray(input.input_images) ||
      input.input_images.some((u) => typeof u !== "string" || !u)
    ) {
      sendSeeapiValidationError(
        res,
        400,
        "Invalid type for parameter: 'input.input_images' must be an array of URLs",
        "invalid_input",
        journal,
        { method, path, headers, body: journalBody },
      );
      return;
    }
  }

  const prompt = typeof input.prompt === "string" ? input.prompt : "";
  const inputImagesCount = Array.isArray(input.input_images) ? input.input_images.length : 0;

  const syntheticReq: ChatCompletionRequest = {
    model,
    messages: [{ role: "user", content: prompt }],
    _endpointType: "seeapi",
    _context: getContext(req),
    _inputImagesCount: inputImagesCount,
  };

  const testId = getTestId(req);
  const idemKey = `${testId}:${idempotencyKey}`;
  const existingId = jobs.getIdempotency(idemKey);
  if (existingId) {
    const existingJob = jobs.get(`${testId}:${existingId}`);
    if (existingJob) {
      if (
        await applyChaosAsync(
          res,
          existingJob.fixture,
          defaults.chaos,
          req.headers,
          req.url,
          journal,
          { method, path, headers, body: syntheticReq, service: "seeapi" },
          existingJob.fixture ? "fixture" : "internal",
          defaults.registry,
          defaults.logger,
        )
      )
        return;
      if (responseGone(res)) return;
      journal.add({
        method,
        path,
        headers,
        body: syntheticReq,
        service: "seeapi",
        response: { status: 202, fixture: existingJob.fixture },
      });
      res.writeHead(202, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ generation: existingJob.submitReply }));
      return;
    }
  }

  const { fixture, skippedBySequenceOrTurn } = matchFixtureDiagnostic(
    fixtures,
    syntheticReq,
    journal.getFixtureMatchCountsForTest(testId),
    defaults.requestTransform,
  );

  if (fixture) {
    journal.incrementFixtureMatchCount(fixture, fixtures, testId);
  }

  const decisions = resolveRequestDecisions(defaults.strict, req.headers);

  if (
    await applyChaosAsync(
      res,
      fixture,
      defaults.chaos,
      req.headers,
      req.url,
      journal,
      { method, path, headers, body: syntheticReq, service: "seeapi" },
      fixture ? "fixture" : "internal",
      defaults.registry,
      defaults.logger,
    )
  )
    return;

  if (!fixture) {
    if (decisions.fixtureMiss) {
      const strictMessage = strictNoMatchMessage(skippedBySequenceOrTurn);
      defaults.logger.error(strictNoMatchLogLine(method, path, skippedBySequenceOrTurn));
      journal.add({
        method,
        path,
        headers,
        body: syntheticReq,
        service: "seeapi",
        response: {
          status: 503,
          fixture: null,
          ...strictOverrideField(defaults.strict, req.headers),
        },
      });
      writeErrorResponse(
        res,
        503,
        renderError("openai", "strict-miss", {
          message: strictMessage,
          type: "invalid_request_error",
          code: "no_fixture_match",
        }).body,
      );
      return;
    }

    journal.add({
      method,
      path,
      headers,
      body: syntheticReq,
      service: "seeapi",
      response: {
        status: 404,
        fixture: null,
        ...strictOverrideField(defaults.strict, req.headers),
      },
    });
    writeErrorResponse(
      res,
      404,
      renderError("openai", "fixture-miss", {
        message: "No fixture matched",
        type: "invalid_request_error",
        code: "no_fixture_match",
      }).body,
    );
    return;
  }

  const worldGeneration = jobs.generation;
  const response = await resolveResponse(fixture, syntheticReq);

  if (isErrorResponse(response)) {
    if (responseGone(res)) return;
    const status = response.status ?? 500;
    journal.add({
      method,
      path,
      headers,
      body: syntheticReq,
      service: "seeapi",
      response: { status, fixture },
    });
    writeErrorResponse(res, status, renderError("openai", "fixture", response.error).body, {
      retryAfter: response.retryAfter,
    });
    return;
  }

  const outcome = extractAsyncImageOutcome(response, SEEAPI_DEFAULT_IMAGE_URL);
  if (!outcome) {
    if (responseGone(res)) return;
    journal.add({
      method,
      path,
      headers,
      body: syntheticReq,
      service: "seeapi",
      response: { status: 500, fixture },
    });
    writeErrorResponse(
      res,
      500,
      renderError("openai", "unknown-response", {
        message: "Fixture response is not an image or JSON type for SeeAPI endpoint",
        type: "server_error",
      }).body,
    );
    return;
  }

  if (fixture.latency && fixture.latency > 0) {
    await delay(calculateDelay(0, undefined, fixture.latency, undefined, defaults.replaySpeed));
  }
  if (responseGone(res)) return;

  const progression = resolveProgression({
    pollsBeforeInProgress: fixture.pollsBeforeInProgress ?? defaults.seeapi?.pollsBeforeInProgress,
    pollsBeforeCompleted: fixture.pollsBeforeCompleted ?? defaults.seeapi?.pollsBeforeCompleted,
  });

  let terminalStatus: "succeeded" | SeeapiTerminalFailureStatus = "succeeded";
  if (outcome.status && SEEAPI_FAILURE_STATUSES.has(outcome.status)) {
    terminalStatus = outcome.status as SeeapiTerminalFailureStatus;
  } else if (outcome.error !== undefined) {
    terminalStatus = "failed";
  }

  const id = crypto.randomUUID();
  const submitReply: SeeapiSubmitReply = {
    id,
    status: "queued",
    ...(outcome.consumedCredits !== undefined ? { consumed_credits: outcome.consumedCredits } : {}),
  };

  const job: SeeapiJob = {
    id,
    model,
    status: progression.pollsBeforeCompleted === 0 ? terminalStatus : "queued",
    terminalStatus,
    urls: outcome.urls,
    errorMessage: outcome.error,
    consumedCredits: outcome.consumedCredits,
    pollCount: 0,
    pollsBeforeInProgress: progression.pollsBeforeInProgress,
    pollsBeforeCompleted: progression.pollsBeforeCompleted,
    fixture,
    submitReply,
  };

  if (jobs.generation === worldGeneration) {
    jobs.set(`${testId}:${id}`, job);
    jobs.setIdempotency(idemKey, id);
  }

  journal.add({
    method,
    path,
    headers,
    body: syntheticReq,
    service: "seeapi",
    response: { status: 202, fixture },
  });

  res.writeHead(202, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ generation: submitReply }));
}

// ─── GET /seeapi/v1/generations/{id} — poll ──────────────────────────────────

export async function handleSeeapiStatus(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  id: string,
  journal: Journal,
  defaults: HandlerDefaults,
  setCorsHeaders: (res: http.ServerResponse) => void,
  jobs: SeeapiJobMap,
): Promise<void> {
  setCorsHeaders(res);
  const path = req.url ?? `/seeapi/v1/generations/${id}`;
  const method = req.method ?? "GET";
  const headers = flattenHeaders(req.headers);

  if (
    await applyChaosAsync(
      res,
      null,
      defaults.chaos,
      req.headers,
      req.url,
      journal,
      { method, path, headers, body: null, service: "seeapi" },
      "internal",
      defaults.registry,
      defaults.logger,
    )
  )
    return;

  if (responseGone(res)) return;

  const testId = getTestId(req);
  const key = `${testId}:${id}`;
  const job = jobs.get(key);

  if (!job) {
    journal.add({
      method,
      path,
      headers,
      body: null,
      service: "seeapi",
      response: { status: 404, fixture: null },
    });
    writeErrorResponse(
      res,
      404,
      renderError("openai", "route-failure", {
        message: `Generation ${id} not found`,
        type: "not_found",
      }).body,
    );
    return;
  }

  advanceSeeapiJob(job);
  jobs.set(key, job);

  journal.add({
    method,
    path,
    headers,
    body: null,
    service: "seeapi",
    response: { status: 200, fixture: null },
  });

  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify(serializeSeeapiPoll(job)));
}

// ─── GET /seeapi/v1/models — static catalog ──────────────────────────────────

export async function handleSeeapiModels(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  journal: Journal,
  defaults: HandlerDefaults,
  setCorsHeaders: (res: http.ServerResponse) => void,
): Promise<void> {
  setCorsHeaders(res);
  const path = req.url ?? "/seeapi/v1/models";
  const method = req.method ?? "GET";
  const headers = flattenHeaders(req.headers);

  if (
    await applyChaosAsync(
      res,
      null,
      defaults.chaos,
      req.headers,
      req.url,
      journal,
      { method, path, headers, body: null, service: "seeapi" },
      "internal",
      defaults.registry,
      defaults.logger,
    )
  )
    return;

  if (responseGone(res)) return;

  journal.add({
    method,
    path,
    headers,
    body: null,
    service: "seeapi",
    response: { status: 200, fixture: null },
  });

  const models = defaults.seeapi?.models ?? DEFAULT_SEEAPI_MODELS;
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ models }));
}
