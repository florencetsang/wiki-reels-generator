/**
 * Read-only API for Wikipedia reels.
 *
 *   GET /reels/<pageid>.json                       reel object from R2
 *   GET /reels/<pageid>/infographics/<NNN>.webp    infographic from R2
 *   GET /related/<pageid>?limit=10&level=3         similar reels via Vectorize
 */

const OBJECT_KEY = /^reels\/\d+(\.json|\/infographics\/\d{3}\.webp)$/;
const RELATED_PATH = /^\/related\/(\d+)$/;
const DEFAULT_RELATED = 10;
// topK is capped at 50 when returning metadata; one slot is used by the reel itself.
const MAX_RELATED = 49;

const CORS = { "Access-Control-Allow-Origin": "*" };

export default {
	async fetch(request, env) {
		if (request.method !== "GET" && request.method !== "HEAD") {
			return json({ error: "Method not allowed" }, 405, { Allow: "GET, HEAD" });
		}

		const url = new URL(request.url);
		const related = url.pathname.match(RELATED_PATH);
		if (related) {
			return relatedReels(related[1], url.searchParams, env);
		}

		const key = url.pathname.slice(1);
		if (OBJECT_KEY.test(key)) {
			return serveObject(key, env);
		}

		return json({ error: "Not found" }, 404);
	},
};

async function relatedReels(pageid, params, env) {
	const limit = params.has("limit") ? Number(params.get("limit")) : DEFAULT_RELATED;
	if (!Number.isInteger(limit) || limit < 1 || limit > MAX_RELATED) {
		return json({ error: `limit must be an integer from 1 to ${MAX_RELATED}` }, 400);
	}

	const options = { topK: limit + 1, returnMetadata: "all" };
	const level = params.get("level");
	if (level !== null) {
		if (!/^[1-5]$/.test(level)) {
			return json({ error: "level must be 1-5" }, 400);
		}
		options.filter = { vital_level: Number(level) };
	}

	const { matches } = await env.VECTORIZE.queryById(pageid, options);
	const related = matches
		.filter((match) => match.id !== pageid)
		.slice(0, limit)
		.map((match) => ({
			pageid: Number(match.id),
			score: match.score,
			...match.metadata,
		}));

	return json({ pageid: Number(pageid), related }, 200, {
		"Cache-Control": "public, max-age=3600",
	});
}

async function serveObject(key, env) {
	const object = await env.REELS.get(key);
	if (object === null) {
		return json({ error: "Not found" }, 404);
	}

	const headers = new Headers(CORS);
	object.writeHttpMetadata(headers);
	headers.set("ETag", object.httpEtag);
	headers.set("Cache-Control", "public, max-age=86400");
	return new Response(object.body, { headers });
}

function json(body, status = 200, headers = {}) {
	return Response.json(body, { status, headers: { ...CORS, ...headers } });
}
