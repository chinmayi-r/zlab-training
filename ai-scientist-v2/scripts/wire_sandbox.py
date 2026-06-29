#!/usr/bin/env python3
"""
wire_sandbox.py - Route AI-Scientist-v2's LLM calls through the Princeton AI Sandbox.

WHY THIS EXISTS (verified against the real repo, commit on `main` as of writing):

AI-Scientist-v2 builds its LLM clients in TWO places, and *both* hardcode a bare
OpenAI client with no base_url hook:

  1. ai_scientist/llm.py :: create_client(model)
        -> for gpt/o1/o3 models it returns `openai.OpenAI()`  (writeup, review,
           citation, ideation, report)
  2. ai_scientist/treesearch/backend/backend_openai.py :: get_ai_client(...)
        -> returns `openai.OpenAI(max_retries=max_retries)`   (the EXPERIMENT
           coding/feedback/vlm models -- i.e. the tree search itself)

The OpenAI Python SDK *does* honour OPENAI_BASE_URL, so a plain OpenAI-compatible
endpoint could be redirected with env vars and no code change. BUT the Princeton
Sandbox is Azure-flavoured: it needs the `AzureOpenAI` client with
`azure_endpoint` + `api_version` (it builds /openai/deployments/<model>/...?api-version=
URLs that the stock OpenAI client does NOT produce). So a base_url alone is not
enough -- a small code patch is genuinely required. This script applies it.

It is idempotent (looks for the marker `# ai-sandbox patch`) and fails loudly if
the repo's source has drifted from the anchors below -- in which case patch the two
functions by hand exactly as shown in report.md.

Gating: the patch only activates when USE_AI_SANDBOX is set in the environment, and
it never touches Claude/Anthropic models (those go through anthropic.* clients that
the Sandbox does not serve). With the Sandbox you must therefore also switch the
experiment model off Bedrock-Claude -- see apply_experiment_config.py --sandbox.

Sandbox facts (verified from PrincetonUniversity/hpc_beginning_workshop myscript.py):
  endpoint     https://api-ai-sandbox.princeton.edu/
  api_version  2025-03-01-preview
  key env var  AI_SANDBOX_KEY
  models       o3-mini, gpt-4o-mini, gpt-4o, gpt-35-turbo-16k,
               Meta-Llama-3-1-70B-Instruct-htzs, Meta-Llama-3-1-8B-Instruct-nwxcg,
               Mistral-small-zgjes
"""
import argparse
import os
import sys

MARKER = "# ai-sandbox patch"

SANDBOX_RETURN = '''    if os.environ.get("USE_AI_SANDBOX") and "claude-" not in model:  {marker}
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=os.environ["AI_SANDBOX_KEY"],
            azure_endpoint=os.environ.get("SANDBOX_ENDPOINT", "https://api-ai-sandbox.princeton.edu/"),
            api_version=os.environ.get("SANDBOX_API_VERSION", "2025-03-01-preview"),
        ), model
'''.format(marker=MARKER)

# For backend_openai.get_ai_client we replace the bare client construction.
# The anchor can sit inside an `else:` block, so the replacement is built with the
# anchor line's own indentation at patch time (see _backend_block).
BACKEND_ANCHOR = "client = openai.OpenAI(max_retries=max_retries)"


def _backend_block(indent):
    """Indentation-aware replacement for the bare-client line in get_ai_client."""
    i = indent
    j = indent + "    "
    return (
        f'{i}if os.environ.get("USE_AI_SANDBOX") and "claude-" not in model:  {MARKER}\n'
        f'{j}from openai import AzureOpenAI\n'
        f'{j}client = AzureOpenAI(\n'
        f'{j}    api_key=os.environ["AI_SANDBOX_KEY"],\n'
        f'{j}    azure_endpoint=os.environ.get("SANDBOX_ENDPOINT", "https://api-ai-sandbox.princeton.edu/"),\n'
        f'{j}    api_version=os.environ.get("SANDBOX_API_VERSION", "2025-03-01-preview"),\n'
        f'{j}    max_retries=max_retries,\n'
        f'{j})\n'
        f'{i}else:\n'
        f'{j}client = openai.OpenAI(max_retries=max_retries)\n'
    )

LLM_ANCHOR = "def create_client(model)"


def _read(path):
    with open(path, "r") as f:
        return f.read()


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _ensure_import_os(text):
    """Make sure the module imports os; prepend if not."""
    for line in text.splitlines()[:40]:
        s = line.strip()
        if s == "import os" or s.startswith("import os "):
            return text
    return "import os\n" + text


def patch_llm(repo):
    path = os.path.join(repo, "ai_scientist", "llm.py")
    if not os.path.isfile(path):
        print(f"  SKIP llm.py: not found at {path}")
        return False
    text = _read(path)
    if MARKER in text:
        print(f"  OK  llm.py already patched")
        return True
    # Find the create_client def line and insert the sandbox short-circuit right after it.
    lines = text.splitlines(keepends=True)
    out, done = [], False
    for i, line in enumerate(lines):
        out.append(line)
        if not done and line.lstrip().startswith(LLM_ANCHOR):
            # Insert after the def line (and after a possible docstring would be safer,
            # but create_client has no docstring in the verified source).
            out.append(SANDBOX_RETURN)
            done = True
    if not done:
        print(f"  FAIL llm.py: could not find `{LLM_ANCHOR}` -- patch manually (see report.md)")
        return False
    text = _ensure_import_os("".join(out))
    _write(path, text)
    print(f"  OK  patched {path}")
    return True


def patch_backend(repo):
    path = os.path.join(repo, "ai_scientist", "treesearch", "backend", "backend_openai.py")
    if not os.path.isfile(path):
        print(f"  SKIP backend_openai.py: not found at {path}")
        return False
    text = _read(path)
    if MARKER in text:
        print(f"  OK  backend_openai.py already patched")
        return True
    lines = text.splitlines(keepends=True)
    out, done = [], False
    for line in lines:
        if not done and line.strip() == BACKEND_ANCHOR:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(_backend_block(indent))
            done = True
        else:
            out.append(line)
    if not done:
        print(f"  FAIL backend_openai.py: anchor not found -- patch manually (see report.md)")
        print(f"       expected line: {BACKEND_ANCHOR}")
        return False
    text = _ensure_import_os("".join(out))
    _write(path, text)
    print(f"  OK  patched {path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.path.expanduser("~/AI-Scientist-v2"),
                    help="Path to the cloned AI-Scientist-v2 repo (default: ~/AI-Scientist-v2)")
    args = ap.parse_args()

    if not os.path.isdir(args.repo):
        sys.exit(f"Repo not found: {args.repo}  (pass --repo /path/to/AI-Scientist-v2)")

    print(f"Wiring Sandbox into {args.repo}")
    a = patch_llm(args.repo)
    b = patch_backend(args.repo)
    print()
    if a and b:
        print("Done. Activate by exporting in your shell / SLURM script:")
        print("  export USE_AI_SANDBOX=1")
        print("  export AI_SANDBOX_KEY=...      # your Sandbox key")
        print("Then point every model flag/config key at a Sandbox model name")
        print("(gpt-4o, o3-mini, gpt-4o-mini, ...). See apply_experiment_config.py --sandbox.")
    else:
        print("One or more patches did not apply. Patch the two functions by hand")
        print("exactly as documented in report.md ('Finding: wiring the Sandbox').")
        sys.exit(1)


if __name__ == "__main__":
    main()
