import * as http from "node:http";
import { describe, test, expect, afterEach } from "vitest";
import { LLMock } from "../llmock.js";
import { startFromConfig } from "../config-loader.js";

async function startWebhookServer(): Promise<{
  url: string;
  received: Array<{ body: unknown }>;
  close: () => Promise<void>;
}> {
  const received: Array<{ body: unknown }> = [];
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      try {
        received.push({ body: JSON.parse(raw) });
      } catch {
        received.push({ body: raw });
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const addr = server.address();
  if (!addr || typeof addr === "string") throw new Error("Unexpected address");
  return {
    url: `http://127.0.0.1:${addr.port}/fal-webhook`,
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
  throw new Error("Timed out waiting for webhook");
}

describe("fal.ai vendor-path and webhook compatibility (opt-in)", () => {
  let mock: LLMock | undefined;
  let closeHooks: Array<() => Promise<void>> = [];

  afterEach(async () => {
    await mock?.stop();
    mock = undefined;
    for (const close of closeHooks) {
      await close();
    }
    closeHooks = [];
  });

  test("default mode (vendorPaths off) ignores /fal/{owner}/{app}/... without x-fal-target-host", async () => {
    mock = new LLMock({ port: 0 });
    mock.onFalImage(/gpt-image-2\.6/, {
      images: [{ url: "https://mock.fal.media/x.png" }],
    });
    await mock.start();

    const res = await fetch(`${mock.url}/fal/openai/gpt-image-2.6/flare/edit`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Key fal-test" },
      body: JSON.stringify({ prompt: "edit" }),
    });
    expect(res.status).toBe(404);
  });

  test("vendorPaths: true serves Provider1 sequence: submit with fal_webhook, 400 while running, 200 on owner/app/requests/{id}, and webhook OK", async () => {
    const webhook = await startWebhookServer();
    closeHooks.push(webhook.close);

    mock = new LLMock({
      port: 0,
      fal: { vendorPaths: true, webhook: true },
      falQueue: { pollsBeforeInProgress: 1, pollsBeforeCompleted: 2 },
    });
    mock.onFalImage("openai/gpt-image-2.6/flare/edit", {
      images: [{ url: "https://mock.fal.media/edited.png" }],
    });
    await mock.start();

    const submitUrl = `${mock.url}/fal/openai/gpt-image-2.6/flare/edit?fal_webhook=${encodeURIComponent(webhook.url)}`;
    const submitRes = await fetch(submitUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Key fal-secret-key",
      },
      body: JSON.stringify({
        prompt: "make it brighter",
        image_urls: ["https://example.com/in.png"],
        image_size: "1024x1024",
        quality: "high",
        output_format: "png",
      }),
    });
    expect(submitRes.status).toBe(200);
    const envelope = await submitRes.json();
    expect(typeof envelope.request_id).toBe("string");
    // status_url and response_url use only owner/app (openai/gpt-image-2.6), NOT /flare/edit
    expect(envelope.status_url).toBe(
      `${mock.url}/fal/openai/gpt-image-2.6/requests/${envelope.request_id}/status`,
    );
    expect(envelope.response_url).toBe(
      `${mock.url}/fal/openai/gpt-image-2.6/requests/${envelope.request_id}`,
    );

    // Poll 1 while running -> answers 400 in vendorPaths mode
    const poll1 = await fetch(
      `${mock.url}/fal/openai/gpt-image-2.6/requests/${envelope.request_id}`,
      { headers: { Authorization: "Key fal-secret-key" } },
    );
    expect(poll1.status).toBe(400);
    expect(webhook.received).toHaveLength(0);

    // Poll 2 reaches COMPLETED -> answers 200 { images: [...] } and delivers OK webhook
    const poll2 = await fetch(
      `${mock.url}/fal/openai/gpt-image-2.6/requests/${envelope.request_id}`,
      { headers: { Authorization: "Key fal-secret-key" } },
    );
    expect(poll2.status).toBe(200);
    const result = await poll2.json();
    expect(result.images).toEqual([
      {
        url: "https://mock.fal.media/edited.png",
        content_type: "image/png",
        width: 1024,
        height: 1024,
      },
    ]);

    const delivered = await waitFor(() => webhook.received[0]);
    expect(delivered.body).toEqual({
      request_id: envelope.request_id,
      status: "OK",
      payload: result,
      error: null,
    });
  });

  test("vendorPaths: true delivers ERROR webhook on failed request and sets image/gif content_type for GIF requests", async () => {
    const webhook = await startWebhookServer();
    closeHooks.push(webhook.close);

    mock = new LLMock({
      port: 0,
      fal: { vendorPaths: true, webhook: true },
    });
    mock.onFalQueue("fal-ai/flux-3-max/sunburst/generate", {
      error: "NSFW content detected",
    });
    mock.onFalImage("openai/gpt-image-2.6/flare/text-to-image", {
      images: [{ url: "https://mock.fal.media/anim" }],
    });
    await mock.start();

    // 1. Error webhook delivery on failed request
    const failSubmit = await fetch(
      `${mock.url}/fal/fal-ai/flux-3-max/sunburst/generate?fal_webhook=${encodeURIComponent(webhook.url)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Key k" },
        body: JSON.stringify({ prompt: "bad" }),
      },
    );
    expect(failSubmit.status).toBe(200);
    const failEnv = await failSubmit.json();
    const errWebhook = await waitFor(() => webhook.received[0]);
    expect(errWebhook.body).toEqual({
      request_id: failEnv.request_id,
      status: "ERROR",
      payload: null,
      error: "NSFW content detected",
    });

    // 2. GIF request returns content_type: "image/gif"
    const gifSubmit = await fetch(`${mock.url}/fal/openai/gpt-image-2.6/flare/text-to-image`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Key k" },
      body: JSON.stringify({
        prompt: "bouncing ball",
        output_format: "gif",
        frame_count: 8,
      }),
    });
    expect(gifSubmit.status).toBe(200);
    const gifEnv = await gifSubmit.json();
    const gifRes = await fetch(
      `${mock.url}/fal/openai/gpt-image-2.6/requests/${gifEnv.request_id}`,
    );
    expect(gifRes.status).toBe(200);
    const gifJson = await gifRes.json();
    expect(gifJson.images[0].content_type).toBe("image/gif");
  });

  test("vendorPaths config loads via startFromConfig and preserves legacy /fal/queue/* routes", async () => {
    const started = await startFromConfig({
      port: 0,
      host: "127.0.0.1",
      fal: { vendorPaths: true, webhook: true },
    });
    mock = started.llmock;
    mock.onFalImage("fal-ai/flux-3-max/text-to-image", {
      images: [{ url: "https://mock.fal.media/flux3.png" }],
    });

    const submit = await fetch(`${started.url}/fal/fal-ai/flux-3-max/text-to-image`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Key k" },
      body: JSON.stringify({ prompt: "space" }),
    });
    expect(submit.status).toBe(200);
    const env = await submit.json();
    const res = await fetch(`${started.url}/fal/fal-ai/flux-3-max/requests/${env.request_id}`);
    expect(res.status).toBe(200);
    expect((await res.json()).images[0].url).toBe("https://mock.fal.media/flux3.png");
  });
});
