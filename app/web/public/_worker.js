// Same-origin API gateway: browsers never need to resolve a tailnet address.
const DEFAULT_UPSTREAM = "https://vmo-namdp15-d.tail09c58b.ts.net";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (!url.pathname.startsWith("/api/")) return env.ASSETS.fetch(request);
    if (!url.pathname.startsWith("/api/v1/") || !["GET", "POST", "OPTIONS"].includes(request.method)) {
      return new Response("Not found", { status: 404 });
    }
    const target = new URL(env.API_UPSTREAM || DEFAULT_UPSTREAM);
    target.pathname = url.pathname.slice(4);
    target.search = url.search;
    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.delete("X-Auscult-Proxy-Token");
    headers.delete("X-Auscult-Client-IP");
    if (env.API_PROXY_TOKEN) {
      headers.set("X-Auscult-Proxy-Token", env.API_PROXY_TOKEN);
      headers.set("X-Auscult-Client-IP", request.headers.get("CF-Connecting-IP") || "unknown");
    }
    try {
      const response = await fetch(target, {
        method: request.method, headers,
        body: request.method === "POST" ? request.body : undefined,
        redirect: "manual",
      });
      const result = new Response(response.body, response);
      result.headers.set("Cache-Control", "no-store");
      return result;
    } catch {
      return Response.json({ error: { code: "upstream_unavailable", message: "The inference API is temporarily unreachable. Please retry." } },
        { status: 502, headers: { "Cache-Control": "no-store" } });
    }
  },
};
