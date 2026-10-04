"""Compressed linux/amd64 size of every Docker image named in a task-set inventory, read without pulling.

Docker Hub images: the Hub web API (hub.docker.com/v2/repositories/<repo>/tags/<tag>/images), which lists the
compressed layers (digest and size) of each architecture and does not count against the Docker Hub pull limit.
Other registries (ghcr.io, aliyuncs.com, public.ecr.aws, ...): the OCI distribution API with an anonymous bearer
token; the layers of the amd64 manifest.
Layer digests are kept so that images sharing layers can be counted once (task_sets_summary.py).

Images considered per task: the prebuilt agent image (`[environment].docker_image`), else the last-stage FROM of
environment/Dockerfile, plus a separate verifier image (`[verifier.environment].docker_image`) when set (or the base of tests/Dockerfile),
plus sidecar images named in environment/docker-compose*.yaml (`image:` lines).

Results are cached in --cache (JSON: image -> {"bytes": int, "layers": {digest: bytes}} or {"error": str});
rerunning only fetches missing ones.

Usage: python3 task_sets_image_sizes.py inventory.json --cache image_sizes.json [--include-bases]
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def parse_ref(ref: str) -> tuple[str, str, str]:
    """Split an image reference into (registry, repository, tag). A digest suffix is dropped; the tag is kept."""
    ref = ref.split("@")[0]
    parts = ref.split("/")
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        registry, rest = parts[0], "/".join(parts[1:])
    else:
        registry, rest = "docker.io", ref
    if ":" in rest.split("/")[-1]:
        repo, tag = rest.rsplit(":", 1)
    else:
        repo, tag = rest, "latest"
    if registry == "docker.io" and "/" not in repo:
        repo = "library/" + repo
    return registry, repo, tag


def http_json(url: str, headers: dict[str, str]) -> tuple[dict, dict]:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read()), dict(resp.headers)


def docker_hub_layers(repo: str, tag: str) -> dict[str, int]:
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}/images"
    while True:
        try:
            data, headers = http_json(url, {})
        except urllib.error.HTTPError as e:
            if e.code == 429:
                reset = int(e.headers.get("x-ratelimit-reset", time.time() + 60))
                time.sleep(max(5, reset - time.time() + 2))
                continue
            raise
        remaining = int(headers.get("x-ratelimit-remaining", "100"))
        if remaining < 5:
            reset = int(headers.get("x-ratelimit-reset", time.time() + 60))
            time.sleep(max(5, reset - time.time() + 2))
        amd64 = [i for i in data if i.get("architecture") == "amd64" and i.get("os") == "linux"]
        if not amd64:
            raise ValueError(f"no linux/amd64 image in Docker Hub tag {repo}:{tag}")
        return {l["digest"]: int(l["size"]) for l in amd64[0]["layers"] if l.get("digest")}


def registry_layers(registry: str, repo: str, tag: str) -> dict[str, int]:
    base = f"https://{registry}/v2/{repo}"
    headers = {"Accept": ACCEPT}
    try:
        manifest, _ = http_json(f"{base}/manifests/{tag}", headers)
    except urllib.error.HTTPError as e:
        challenge = e.headers.get("www-authenticate", "")
        if e.code != 401 or not challenge.lower().startswith("bearer"):
            raise
        fields = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
        token_url = f"{fields['realm']}?service={fields.get('service', '')}&scope=repository:{repo}:pull"
        token_data, _ = http_json(token_url, {})
        headers["Authorization"] = "Bearer " + (token_data.get("token") or token_data["access_token"])
        manifest, _ = http_json(f"{base}/manifests/{tag}", headers)
    if "manifests" in manifest:
        amd64 = [
            m
            for m in manifest["manifests"]
            if m.get("platform", {}).get("architecture") == "amd64" and m.get("platform", {}).get("os") == "linux"
        ]
        if not amd64:
            raise ValueError(f"no linux/amd64 manifest in {registry}/{repo}:{tag}")
        manifest, _ = http_json(f"{base}/manifests/{amd64[0]['digest']}", headers)
    return {layer["digest"]: int(layer["size"]) for layer in manifest["layers"]}


def image_layers(ref: str) -> dict[str, int]:
    registry, repo, tag = parse_ref(ref)
    if registry in ("docker.io", "registry-1.docker.io", "index.docker.io"):
        return docker_hub_layers(repo, tag)
    return registry_layers(registry, repo, tag)


def images_of(row: dict, include_bases: bool) -> list[str]:
    out = [row["prebuilt_image"] or row["dockerfile_base"]]
    if include_bases and row["dockerfile_base"]:
        # Also the Dockerfile base of prebuilt-image tasks: used to calibrate the size estimate for Dockerfile-only tasks.
        out.append(row["dockerfile_base"])
    if row["verifier_image"]:
        out.append(row["verifier_image"])
    if row["verifier_dockerfile_base"]:
        out.append(row["verifier_dockerfile_base"])
    out.extend(row["compose_images"])
    return [i for i in out if i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--include-bases", action="store_true")
    args = ap.parse_args()
    rows = json.loads(args.inventory.read_text())
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    wanted = sorted({img for r in rows for img in images_of(r, args.include_bases)})
    todo = [i for i in wanted if i not in cache or ("bytes" in cache[i] and "layers" not in cache[i])]
    print(f"{len(wanted)} images, {len(todo)} to fetch", file=sys.stderr)
    for n, img in enumerate(todo, 1):
        try:
            layers = image_layers(img)
            cache[img] = {"bytes": sum(layers.values()), "layers": layers}
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, KeyError, TimeoutError) as e:
            # Recorded, not hidden: the report lists every image whose size could not be read.
            cache[img] = {"error": f"{type(e).__name__}: {e}"}
        if n % 25 == 0:
            args.cache.write_text(json.dumps(cache, indent=1, sort_keys=True))
            print(f"  {n}/{len(todo)}", file=sys.stderr)
    args.cache.write_text(json.dumps(cache, indent=1, sort_keys=True))
    errors = [i for i in wanted if "error" in cache[i]]
    print(f"done; {len(errors)} errors", file=sys.stderr)
    for i in errors:
        print(f"  {i}: {cache[i]['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
