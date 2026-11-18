import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import { describe, test, expect, afterEach } from "vitest";
import { LLMock } from "../llmock.js";
import { startFromConfig } from "../config-loader.js";
import { KieJobMap, KIE_MAX_ENTRIES } from "../kie.js";

async function startCallbackServer(): Promise<{
  url: string;
  received: Array<{ body: unknown; headers: http.IncomingHttpHeaders }>;
  close: () => Promise<void>;
}> {
  const received: Array<{ body: unknown; headers: http.IncomingHttpHeaders }> = [];
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      try {
        received.push({ body: JSON.parse(raw), headers: req.headers });
      } catch {
        received.push({ body: raw, headers: req.headers });
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const addr = server.address();
  if (!addr || typeof addr === "string") throw new Error("Unexpected address");
  return {
    url: `http://127.0.0.1:${addr.port}/callback`,
    received,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

async function waitFor<T>(fn: () => T | undefined, timeoutMs = 3000): Promise<T> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const val = fn();
    if (val !== undefined) return val;
    await new Promise((r) => setTimeout(r, 20));
  }
  throw new Error("Timed out waiting for condition");
}

// ─── KieJobMap unit surface ──────────────────────────────────────────────────

describe("KieJobMap", () => {
  test("is exported with the bounded-entries constant and clear resets generation", () => {
    expect(KIE_MAX_ENTRIES).toBe(10_000);
    const map = new KieJobMap();
    expect(map.size).toBe(0);
    expect(map.generation).toBe(0);
    map.clear();
    expect(map.generation).toBe(1);
  });
});

// ─── KIE HTTP lifecycle (/kie/api/v1/...) ─────────────────────────────────────

describe("KIE mock (/kie/api/v1/...)", () => {
  let mock: LLMock | undefined;
  let closeHooks: Array<() => Promise<void>> = [];
  let tmpDirs: string[] = [];

  afterEach(async () => {
    await mock?.stop();
    mock = undefined;
    for (const close of closeHooks) {
      await close();
    }
    closeHooks = [];
    for (const dir of tmpDirs) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
    tmpDirs = [];
  });

  test("createTask and recordInfo return KIE envelope with resultJson as a JSON string", async () => {
    mock = new LLMock({ port: 0 });
    mock.onKieImage("nano-banana-2", {
      urls: ["https://tempfile.aiquickdraw.com/out/dog.png"],
    });
    await mock.start();

    const submit = await fetch(`${mock.url}/kie/api/v1/jobs/createTask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer kie-test-key",
      },
      body: JSON.stringify({
        model: "nano-banana-2",
        input: {
          prompt: "a happy golden retriever",
          aspect_ratio: "16:9",
          resolution: "2K",
        },
      }),
    });
    expect(submit.status).toBe(200);
    const submitBody = await submit.json();
    expect(submitBody.code).toBe(200);
    expect(submitBody.msg).toBe("success");
    const taskId: string = submitBody.data.taskId;
    expect(typeof taskId).toBe("string");
    expect(taskId.length).toBeGreaterThan(0);

    const poll = await fetch(`${mock.url}/kie/api/v1/jobs/recordInfo?taskId=${taskId}`, {
      headers: { Authorization: "Bearer kie-test-key" },
    });
    expect(poll.status).toBe(200);
    const pollBody = await poll.json();
    expect(pollBody.code).toBe(200);
    expect(pollBody.msg).toBe("success");
    expect(pollBody.data.taskId).toBe(taskId);
    expect(pollBody.data.model).toBe("nano-banana-2");
    expect(pollBody.data.state).toBe("success");
    // Explicit assertion for the KIE resultJson string trap:
    // resultJson MUST be a JSON-encoded string, NOT a parsed object.
    expect(typeof pollBody.data.resultJson).toBe("string");
    expect(JSON.parse(pollBody.data.resultJson)).toEqual({
      resultUrls: ["https://tempfile.aiquickdraw.com/out/dog.png"],
    });
    expect(pollBody.data.failCode).toBeNull();
    expect(pollBody.data.failMsg).toBeNull();
    expect(typeof pollBody.data.createTime).toBe("number");
    expect(typeof pollBody.data.completeTime).toBe("number");
    expect(typeof pollBody.data.costTime).toBe("number");
  });

  test("uses /kie prefix and does not collide with /api/v1/jobs/createTask", async () => {
    mock = new LLMock({ port: 0 });
    mock.onKieImage("nano-banana-2", { urls: ["https://example.com/x.png"] });
    await mock.start();

    const unprefixed = await fetch(`${mock.url}/api/v1/jobs/createTask`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer k" },
      body: JSON.stringify({ model: "nano-banana-2", input: { prompt: "x" } }),
    });
    expect(unprefixed.status).toBe(404);
  });

  test("missing Authorization returns 401 { code: 401, msg } and bad body returns 422 { code: 422, msg }", async () => {
    mock = new LLMock({ port: 0 });
    mock.onKieImage("nano-banana-2", { urls: ["https://example.com/x.png"] });
    await mock.start();

    const noAuth = await fetch(`${mock.url}/kie/api/v1/jobs/createTask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "nano-banana-2", input: { prompt: "hi" } }),
    });
    expect(noAuth.status).toBe(401);
    const noAuthBody = await noAuth.json();
    expect(noAuthBody.code).toBe(401);
    expect(typeof noAuthBody.msg).toBe("string");

    const badJson = await fetch(`${mock.url}/kie/api/v1/jobs/createTask`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer k" },
      body: "{bad",
    });
    expect(badJson.status).toBe(422);
    expect((await badJson.json()).code).toBe(422);

    const missingInput = await fetch(`${mock.url}/kie/api/v1/jobs/createTask`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer k" },
      body: JSON.stringify({ model: "nano-banana-2" }),
    });
    expect(missingInput.status).toBe(422);
    expect((await missingInput.json()).code).toBe(422);

    const missingTaskId = await fetch(`${mock.url}/kie/api/v1/jobs/recordInfo`, {
      headers: { Authorization: "Bearer k" },
    });
    expect(missingTaskId.status).toBe(422);
    expect((await missingTaskId.json()).code).toBe(422);
  });

  test("poll progression transitions queuing -> generating -> success", async () => {
    mock = new LLMock({
      port: 0,
      kie: { pollsBeforeInProgress: 2, pollsBeforeCompleted: 3 },
    });
    mock.onKieImage(/landscape/, { urls: ["https://tempfile.aiquickdraw.com/l.png"] });
    await mock.start();

    const submit = await (
      await fetch(`${mock.url}/kie/api/v1/jobs/createTask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer k" },
        body: JSON.stringify({ model: "nano-banana-2", input: { prompt: "mountain landscape" } }),
      })
    ).json();
    const taskId: string = submit.data.taskId;

    const p1 = await (
      await fetch(`${mock.url}/kie/api/v1/jobs/recordInfo?taskId=${taskId}`, {
        headers: { Authorization: "Bearer k" },
      })
    ).json();
    expect(p1.data.state).toBe("queuing");
    expect(p1.data.resultJson).toBe("");

    const p2 = await (
      await fetch(`${mock.url}/kie/api/v1/jobs/recordInfo?taskId=${taskId}`, {
        headers: { Authorization: "Bearer k" },
      })
    ).json();
    expect(p2.data.state).toBe("generating");

    const p3 = await (
      await fetch(`${mock.url}/kie/api/v1/jobs/recordInfo?taskId=${taskId}`, {
        headers: { Authorization: "Bearer k" },
      })
    ).json();
    expect(p3.data.state).toBe("success");
    expect(typeof p3.data.resultJson).toBe("string");
    expect(JSON.parse(p3.data.resultJson)).toEqual({
      resultUrls: ["https://tempfile.aiquickdraw.com/l.png"],
    });
  });

  test("failure fixture polls to state='fail' with failMsg and delivers callback once", async () => {
    const cb = await startCallbackServer();
    closeHooks.push(cb.close);

    mock = new LLMock({
      port: 0,
      kie: { pollsBeforeInProgress: 1, pollsBeforeCompleted: 2 },
    });
    mock.onKieImage(
      "nano-banana-2",
      { error: "prompt blocked", failCode: "422" },
      { prompt: "bad prompt" },
    );
    await mock.start();

    const submit = await (
      await fetch(`${mock.url}/kie/api/v1/jobs/createTask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer k" },
        body: JSON.stringify({
          model: "nano-banana-2",
          input: { prompt: "bad prompt" },
          callBackUrl: cb.url,
        }),
      })
    ).json();
    const taskId = submit.data.taskId;

    const p1 = await (
      await fetch(`${mock.url}/kie/api/v1/jobs/recordInfo?taskId=${taskId}`, {
        headers: { Authorization: "Bearer k" },
      })
    ).json();
    expect(p1.data.state).toBe("generating");
    expect(cb.received).toHaveLength(0);

    const p2 = await (
      await fetch(`${mock.url}/kie/api/v1/jobs/recordInfo?taskId=${taskId}`, {
        headers: { Authorization: "Bearer k" },
      })
    ).json();
    expect(p2.data.state).toBe("fail");
    expect(p2.data.failMsg).toBe("prompt blocked");
    expect(p2.data.failCode).toBe("422");

    const delivered = await waitFor(() => cb.received[0]);
    const payload = delivered.body as {
      code: number;
      data: { taskId: string; state: string; failMsg: string };
    };
    expect(payload.code).toBe(200);
    expect(payload.data.taskId).toBe(taskId);
    expect(payload.data.state).toBe("fail");
    expect(payload.data.failMsg).toBe("prompt blocked");

    // Polling again must not send a duplicate callback
    await fetch(`${mock.url}/kie/api/v1/jobs/recordInfo?taskId=${taskId}`, {
      headers: { Authorization: "Bearer k" },
    });
    await new Promise((r) => setTimeout(r, 50));
    expect(cb.received).toHaveLength(1);
  });

  test("callBackUrl delivers immediately on 0/0 completion, tolerates unreachable URLs, and is switchable", async () => {
    const cb = await startCallbackServer();
    closeHooks.push(cb.close);

    mock = new LLMock({ port: 0 });
    mock.onKieImage("nano-banana-2", {
      urls: ["https://tempfile.aiquickdraw.com/cb.png"],
    });
    await mock.start();

    const submit = await (
      await fetch(`${mock.url}/kie/api/v1/jobs/createTask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer k" },
        body: JSON.stringify({
          model: "nano-banana-2",
          input: { prompt: "cb success" },
          callBackUrl: cb.url,
        }),
      })
    ).json();

    const delivered = await waitFor(() => cb.received[0]);
    const payload = delivered.body as {
      code: number;
      data: { taskId: string; state: string; resultJson: string };
    };
    expect(payload.code).toBe(200);
    expect(payload.data.taskId).toBe(submit.data.taskId);
    expect(payload.data.state).toBe("success");
    expect(typeof payload.data.resultJson).toBe("string");
    expect(JSON.parse(payload.data.resultJson)).toEqual({
      resultUrls: ["https://tempfile.aiquickdraw.com/cb.png"],
    });

    // Unreachable callback URL must never crash
    const unreachable = await fetch(`${mock.url}/kie/api/v1/jobs/createTask`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer k" },
      body: JSON.stringify({
        model: "nano-banana-2",
        input: { prompt: "unreachable cb" },
        callBackUrl: "http://127.0.0.1:1/unreachable-callback",
      }),
    });
    expect(unreachable.status).toBe(200);
    await mock.stop();

    // Switchable callbacks: false
    const cb2 = await startCallbackServer();
    closeHooks.push(cb2.close);
    mock = new LLMock({ port: 0, kie: { callbacks: false } });
    mock.onKieImage("nano-banana-2", { urls: ["https://example.com/no-cb.png"] });
    await mock.start();

    await fetch(`${mock.url}/kie/api/v1/jobs/createTask`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer k" },
      body: JSON.stringify({
        model: "nano-banana-2",
        input: { prompt: "disabled cb" },
        callBackUrl: cb2.url,
      }),
    });
    await new Promise((r) => setTimeout(r, 60));
    expect(cb2.received).toHaveLength(0);
  });

  test("matches on image_input presence/count, journals service='kie', and applies chaos", async () => {
    mock = new LLMock({ port: 0 });
    mock.onKieImage(
      "nano-banana-2",
      { urls: ["https://tempfile.aiquickdraw.com/edit.png"] },
      { hasInputImages: true, inputImageCount: 1 },
    );
    await mock.start();

    const submit = await (
      await fetch(`${mock.url}/kie/api/v1/jobs/createTask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer k" },
        body: JSON.stringify({
          model: "nano-banana-2",
          input: {
            prompt: "edit photo",
            image_input: ["https://example.com/source.png"],
          },
        }),
      })
    ).json();
    const poll = await (
      await fetch(`${mock.url}/kie/api/v1/jobs/recordInfo?taskId=${submit.data.taskId}`, {
        headers: { Authorization: "Bearer k" },
      })
    ).json();
    expect(JSON.parse(poll.data.resultJson)).toEqual({
      resultUrls: ["https://tempfile.aiquickdraw.com/edit.png"],
    });

    const chaosDrop = await fetch(`${mock.url}/kie/api/v1/jobs/createTask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer k",
        "X-AIMock-Chaos-Drop": "1",
      },
      body: JSON.stringify({ model: "nano-banana-2", input: { prompt: "chaos" } }),
    });
    expect(chaosDrop.status).toBe(500);

    const journalRes = await fetch(`${mock.url}/__aimock/journal?service=kie`);
    const journalBody = await journalRes.json();
    expect(Array.isArray(journalBody)).toBe(true);
    expect(journalBody.length).toBe(3);
    expect(journalBody.every((e: { service?: string }) => e.service === "kie")).toBe(true);

    mock.reset();
    const afterReset = await fetch(
      `${mock.url}/kie/api/v1/jobs/recordInfo?taskId=${submit.data.taskId}`,
      {
        headers: { Authorization: "Bearer k" },
      },
    );
    expect(afterReset.status).toBe(404);
  });

  test("loads from config file and JSON fixture file", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "aimock-kie-cfg-"));
    tmpDirs.push(dir);
    const fixturePath = path.join(dir, "fixtures.json");
    fs.writeFileSync(
      fixturePath,
      JSON.stringify({
        fixtures: [
          {
            match: { endpoint: "kie", model: "nano-banana-2", userMessage: "from file" },
            response: { urls: ["https://tempfile.aiquickdraw.com/file.png"] },
          },
        ],
      }),
    );

    const started = await startFromConfig({
      port: 0,
      host: "127.0.0.1",
      llm: { fixtures: fixturePath },
      kie: { callbacks: true },
    });
    mock = started.llmock;

    const submit = await (
      await fetch(`${started.url}/kie/api/v1/jobs/createTask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer k" },
        body: JSON.stringify({ model: "nano-banana-2", input: { prompt: "from file" } }),
      })
    ).json();
    const poll = await (
      await fetch(`${started.url}/kie/api/v1/jobs/recordInfo?taskId=${submit.data.taskId}`, {
        headers: { Authorization: "Bearer k" },
      })
    ).json();
    expect(poll.data.state).toBe("success");
    expect(JSON.parse(poll.data.resultJson)).toEqual({
      resultUrls: ["https://tempfile.aiquickdraw.com/file.png"],
    });
  });
});
