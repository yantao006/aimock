import type * as http from "node:http";
import crypto from "node:crypto";
import type { ChatCompletionRequest, Fixture, HandlerDefaults } from "./types.js";
import {
  flattenHeaders,
  getContext,
  getTestId,
  isErrorResponse,
  isJsonObject,
  resolveResponse,
  strictNoMatchLogLine,
  strictNoMatchMessage,
} from "./helpers.js";
import { resolveRequestDecisions, strictOverrideField } from "./response/policy.js";
import { matchFixtureDiagnostic } from "./router.js";
import { calculateDelay, delay } from "./sse-writer.js";
import type { Journal } from "./journal.js";
import { applyChaosAsync, responseGone } from "./chaos.js";
import { resolveProgression } from "./fal.js";
import { extractAsyncImageOutcome } from "./seeapi.js";
import type { Logger } from "./logger.js";

/**
 * KIE async image generation mock.
 *
 * - Submit: `POST /kie/api/v1/jobs/createTask` (Authorization: Bearer <key>,
 *   JSON body `{ model, input: { prompt, image_input?, ... }, callBackUrl? }`)
 *   -> HTTP 200 `{ code: 200, msg: "success", data: { taskId } }`.
 * - Poll: `GET /kie/api/v1/jobs/recordInfo?taskId=<id>` -> HTTP 200
 *   `{ code: 200, msg: "success", data: { taskId, model, state, resultJson, failCode, failMsg, costTime, createTime, completeTime } }`.
 * - Note: `resultJson` is always a JSON-encoded string (e.g.
 *   `"{\"resultUrls\":[\"https://...\"]}"`), matching real KIE wire behavior.
 * - Callback: when `callBackUrl` is provided and callbacks are enabled, POSTs
 *   the terminal `recordInfo` payload once (tolerant of unreachable URLs).
 */

export const KIE_MAX_ENTRIES = 10_000;
const KIE_TTL_MS = 3_600_000; // 1 hour
const KIE_CALLBACK_TIMEOUT_MS = 5_000;

const KIE_DEFAULT_IMAGE_URL = "https://tempfile.aiquickdraw.com/aimock-placeholder-image.png";

export type KieTaskState = "waiting" | "queuing" | "generating" | "success" | "fail";

export interface KieJob {
  taskId: string;
  model: string;
  state: KieTaskState;
  terminalState: "success" | "fail";
  urls: string[];
  failCode: string | null;
  failMsg: string | null;
  createTime: number;
  completeTime: number | null;
  costTime: number | null;
  callBackUrl?: string;
  callbacksEnabled: boolean;
  callbackSent: boolean;
  pollCount: number;
  pollsBeforeInProgress: number;
  pollsBeforeCompleted: number;
}

interface KieEntry {
  job: KieJob;
  createdAt: number;
}

/**
 * Bounded per-testId job store for KIE tasks.
 */
export class KieJobMap {
  private readonly entries = new Map<string, KieEntry>();
  private worldGeneration = 0;

  get generation(): number {
    return this.worldGeneration;
  }

  get(key: string): KieJob | undefined {
    const entry = this.entries.get(key);
    if (!entry) return undefined;
    if (Date.now() - entry.createdAt > KIE_TTL_MS) {
      this.entries.delete(key);
      return undefined;
    }
    return entry.job;
  }

  set(key: string, job: KieJob): void {
    this.entries.delete(key);
    this.entries.set(key, { job, createdAt: Date.now() });
    if (this.entries.size > KIE_MAX_ENTRIES) {
      const excess = this.entries.size - KIE_MAX_ENTRIES;
      const iter = this.entries.keys();
      for (let i = 0; i < excess; i++) {
        const next = iter.next();
        if (!next.done) this.entries.delete(next.value);
      }
    }
  }

  delete(key: string): boolean {
    return this.entries.delete(key);
  }

  clear(): void {
    this.entries.clear();
    this.worldGeneration++;
  }

  get size(): number {
    return this.entries.size;
  }
}

function headerSingle(headers: http.IncomingHttpHeaders, name: string): string | undefined {
  const val = headers[name];
  if (Array.isArray(val)) return val[0];
  return val;
}

function hasValidBearerAuth(req: http.IncomingMessage): boolean {
  const auth = headerSingle(req.headers, "authorization")?.trim() ?? "";
  return /^Bearer\s+\S+$/i.test(auth);
}

function writeKieReply(
  res: http.ServerResponse,
  httpStatus: number,
  body: Record<string, unknown>,
): void {
  res.writeHead(httpStatus, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

function writeKieError(
  res: http.ServerResponse,
  httpStatus: number,
  code: number,
  msg: string,
  journal: Journal,
  meta: {
    method: string;
    path: string;
    headers: Record<string, string>;
    body: ChatCompletionRequest | null;
    extra?: Record<string, unknown>;
  },
): void {
  journal.add({
    method: meta.method,
    path: meta.path,
    headers: meta.headers,
    body: meta.body,
    service: "kie",
    response: { status: httpStatus, fixture: null, ...meta.extra },
  });
  writeKieReply(res, httpStatus, { code, msg });
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

export function buildKieRecordInfoPayload(job: KieJob): Record<string, unknown> {
  const isSuccess = job.state === "success";
  const isFail = job.state === "fail";
  return {
    code: 200,
    msg: "success",
    data: {
      taskId: job.taskId,
      model: job.model,
      state: job.state,
      resultJson: isSuccess ? JSON.stringify({ resultUrls: job.urls }) : "",
      failCode: isFail ? (job.failCode ?? "500") : null,
      failMsg: isFail ? (job.failMsg ?? "Generation failed") : null,
      costTime: job.costTime,
      createTime: job.createTime,
      completeTime: job.completeTime,
    },
  };
}

function advanceKieJob(job: KieJob): boolean {
  if (job.state === "success" || job.state === "fail") return false;
  job.pollCount += 1;
  if (job.pollCount >= job.pollsBeforeCompleted) {
    job.state = job.terminalState;
    const now = Date.now();
    job.completeTime = now;
    job.costTime = Math.max(1, now - job.createTime);
    return true;
  }
  if (job.pollCount >= job.pollsBeforeInProgress) {
    job.state = "generating";
  } else {
    job.state = "queuing";
  }
  return false;
}

async function deliverKieCallback(job: KieJob, logger: Logger): Promise<void> {
  if (!job.callBackUrl || !job.callbacksEnabled || job.callbackSent) return;
  job.callbackSent = true;
  const payload = buildKieRecordInfoPayload(job);
  try {
    const res = await fetch(job.callBackUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(KIE_CALLBACK_TIMEOUT_MS),
    });
    if (!res.ok) {
      logger.warn(
        `KIE callback to ${job.callBackUrl} for task ${job.taskId} returned HTTP ${res.status}`,
      );
    }
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    logger.warn(`KIE callback to ${job.callBackUrl} for task ${job.taskId} failed: ${detail}`);
  }
}

function isValidHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

// ─── POST /kie/api/v1/jobs/createTask — submit ───────────────────────────────

export async function handleKieCreateTask(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  raw: string,
  fixtures: Fixture[],
  journal: Journal,
  defaults: HandlerDefaults,
  setCorsHeaders: (res: http.ServerResponse) => void,
  jobs: KieJobMap,
): Promise<void> {
  setCorsHeaders(res);
  const path = req.url ?? "/kie/api/v1/jobs/createTask";
  const method = req.method ?? "POST";
  const headers = flattenHeaders(req.headers);

  if (defaults.kie?.requireAuth !== false && !hasValidBearerAuth(req)) {
    writeKieError(
      res,
      401,
      401,
      "Unauthorized: missing or invalid Authorization Bearer token",
      journal,
      {
        method,
        path,
        headers,
        body: null,
      },
    );
    return;
  }

  const contentType = headerSingle(req.headers, "content-type") ?? "";
  if (contentType.toLowerCase().includes("multipart/form-data")) {
    writeKieError(res, 422, 422, "multipart/form-data is not supported", journal, {
      method,
      path,
      headers,
      body: null,
    });
    return;
  }

  let parsedRaw: unknown;
  try {
    parsedRaw = JSON.parse(raw);
  } catch (parseErr) {
    const detail = parseErr instanceof Error ? parseErr.message : "unknown";
    writeKieError(res, 422, 422, `Malformed JSON: ${detail}`, journal, {
      method,
      path,
      headers,
      body: null,
    });
    return;
  }

  if (!isJsonObject(parsedRaw)) {
    writeKieError(res, 422, 422, "Request body must be a JSON object", journal, {
      method,
      path,
      headers,
      body: null,
    });
    return;
  }

  const journalBody = validationJournalBody(parsedRaw);

  if (typeof parsedRaw.model !== "string" || !parsedRaw.model.trim()) {
    writeKieError(res, 422, 422, "Missing or invalid parameter: 'model'", journal, {
      method,
      path,
      headers,
      body: journalBody,
    });
    return;
  }
  const model = parsedRaw.model.trim();

  if (!isJsonObject(parsedRaw.input)) {
    writeKieError(res, 422, 422, "Missing or invalid parameter: 'input'", journal, {
      method,
      path,
      headers,
      body: journalBody,
    });
    return;
  }
  const input = parsedRaw.input;

  if (input.prompt !== undefined && typeof input.prompt !== "string") {
    writeKieError(res, 422, 422, "Invalid parameter: 'input.prompt' must be a string", journal, {
      method,
      path,
      headers,
      body: journalBody,
    });
    return;
  }

  if (input.image_input !== undefined) {
    if (
      !Array.isArray(input.image_input) ||
      input.image_input.some((u) => typeof u !== "string" || !u)
    ) {
      writeKieError(
        res,
        422,
        422,
        "Invalid parameter: 'input.image_input' must be an array of URLs",
        journal,
        { method, path, headers, body: journalBody },
      );
      return;
    }
  }

  let callBackUrl: string | undefined;
  if (
    parsedRaw.callBackUrl !== undefined &&
    parsedRaw.callBackUrl !== null &&
    parsedRaw.callBackUrl !== ""
  ) {
    if (typeof parsedRaw.callBackUrl !== "string" || !isValidHttpUrl(parsedRaw.callBackUrl)) {
      writeKieError(
        res,
        422,
        422,
        "Invalid parameter: 'callBackUrl' must be an http/https URL",
        journal,
        { method, path, headers, body: journalBody },
      );
      return;
    }
    callBackUrl = parsedRaw.callBackUrl;
  }

  const prompt = typeof input.prompt === "string" ? input.prompt : "";
  const inputImagesCount = Array.isArray(input.image_input) ? input.image_input.length : 0;

  const syntheticReq: ChatCompletionRequest = {
    model,
    messages: [{ role: "user", content: prompt }],
    _endpointType: "kie",
    _context: getContext(req),
    _inputImagesCount: inputImagesCount,
  };

  const testId = getTestId(req);
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
      { method, path, headers, body: syntheticReq, service: "kie" },
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
      writeKieError(res, 503, 503, strictMessage, journal, {
        method,
        path,
        headers,
        body: syntheticReq,
        extra: strictOverrideField(defaults.strict, req.headers),
      });
      return;
    }

    writeKieError(res, 404, 404, "No fixture matched", journal, {
      method,
      path,
      headers,
      body: syntheticReq,
      extra: strictOverrideField(defaults.strict, req.headers),
    });
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
      service: "kie",
      response: { status, fixture },
    });
    writeKieReply(res, status, { code: status, msg: response.error.message });
    return;
  }

  const outcome = extractAsyncImageOutcome(response, KIE_DEFAULT_IMAGE_URL);
  if (!outcome) {
    if (responseGone(res)) return;
    journal.add({
      method,
      path,
      headers,
      body: syntheticReq,
      service: "kie",
      response: { status: 500, fixture },
    });
    writeKieReply(res, 500, {
      code: 500,
      msg: "Fixture response is not an image or JSON type for KIE endpoint",
    });
    return;
  }

  if (fixture.latency && fixture.latency > 0) {
    await delay(calculateDelay(0, undefined, fixture.latency, undefined, defaults.replaySpeed));
  }
  if (responseGone(res)) return;

  const progression = resolveProgression({
    pollsBeforeInProgress: fixture.pollsBeforeInProgress ?? defaults.kie?.pollsBeforeInProgress,
    pollsBeforeCompleted: fixture.pollsBeforeCompleted ?? defaults.kie?.pollsBeforeCompleted,
  });

  const isFailure =
    outcome.error !== undefined ||
    (outcome.status !== undefined &&
      outcome.status !== "succeeded" &&
      outcome.status !== "queued" &&
      outcome.status !== "processing");

  const terminalState: "success" | "fail" = isFailure ? "fail" : "success";
  const now = Date.now();
  const taskId = crypto.randomUUID();
  const immediate = progression.pollsBeforeCompleted === 0;

  const job: KieJob = {
    taskId,
    model,
    state: immediate ? terminalState : "waiting",
    terminalState,
    urls: outcome.urls,
    failCode: isFailure ? (outcome.failCode ?? "500") : null,
    failMsg: isFailure ? (outcome.error ?? "Generation failed") : null,
    createTime: now,
    completeTime: immediate ? now : null,
    costTime: immediate ? 0 : null,
    callBackUrl,
    callbacksEnabled: defaults.kie?.callbacks !== false,
    callbackSent: false,
    pollCount: 0,
    pollsBeforeInProgress: progression.pollsBeforeInProgress,
    pollsBeforeCompleted: progression.pollsBeforeCompleted,
  };

  if (jobs.generation === worldGeneration) {
    jobs.set(`${testId}:${taskId}`, job);
  }

  journal.add({
    method,
    path,
    headers,
    body: syntheticReq,
    service: "kie",
    response: { status: 200, fixture },
  });

  writeKieReply(res, 200, {
    code: 200,
    msg: "success",
    data: { taskId },
  });

  if (immediate && job.callBackUrl && job.callbacksEnabled) {
    setImmediate(() => {
      void deliverKieCallback(job, defaults.logger);
    });
  }
}

// ─── GET /kie/api/v1/jobs/recordInfo?taskId=<id> — poll ──────────────────────

export async function handleKieRecordInfo(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  journal: Journal,
  defaults: HandlerDefaults,
  setCorsHeaders: (res: http.ServerResponse) => void,
  jobs: KieJobMap,
): Promise<void> {
  setCorsHeaders(res);
  const path = req.url ?? "/kie/api/v1/jobs/recordInfo";
  const method = req.method ?? "GET";
  const headers = flattenHeaders(req.headers);

  if (defaults.kie?.requireAuth !== false && !hasValidBearerAuth(req)) {
    writeKieError(
      res,
      401,
      401,
      "Unauthorized: missing or invalid Authorization Bearer token",
      journal,
      {
        method,
        path,
        headers,
        body: null,
      },
    );
    return;
  }

  if (
    await applyChaosAsync(
      res,
      null,
      defaults.chaos,
      req.headers,
      req.url,
      journal,
      { method, path, headers, body: null, service: "kie" },
      "internal",
      defaults.registry,
      defaults.logger,
    )
  )
    return;

  if (responseGone(res)) return;

  const url = new URL(req.url ?? "/kie/api/v1/jobs/recordInfo", "http://localhost");
  const taskId = url.searchParams.get("taskId")?.trim() ?? "";
  if (!taskId) {
    writeKieError(res, 422, 422, "Missing required query parameter: taskId", journal, {
      method,
      path,
      headers,
      body: null,
    });
    return;
  }

  const testId = getTestId(req);
  const key = `${testId}:${taskId}`;
  const job = jobs.get(key);

  if (!job) {
    writeKieError(res, 404, 404, `Task ${taskId} not found`, journal, {
      method,
      path,
      headers,
      body: null,
    });
    return;
  }

  const reachedTerminal = advanceKieJob(job);
  jobs.set(key, job);

  journal.add({
    method,
    path,
    headers,
    body: null,
    service: "kie",
    response: { status: 200, fixture: null },
  });

  writeKieReply(res, 200, buildKieRecordInfoPayload(job));

  if (reachedTerminal && job.callBackUrl && job.callbacksEnabled) {
    setImmediate(() => {
      void deliverKieCallback(job, defaults.logger);
    });
  }
}
