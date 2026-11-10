# EmberBridge LIBERO agent

This evaluation-level adapter lets Gym collect complete LIBERO SDK episodes. Gym's
model server owns provider credentials and request capture. EmberBridge owns the
actor loop, Nanoverse owns the SDK and simulator, and the source owner's native
verification decides the result. The `legacy_agent` environment relay connects
this `/run` integration to Gym's collector. It does not implement the agent session
or continuation protocol.

The initial workload is [libero_sdk](../../environments/libero_sdk/README.md).
It requires access to an operator-selected source repository containing sibling
`ember_bridge` and `nanoverse` components. Neither private project code nor a
simulator is redistributed with Gym. Source and asset licensing and accessibility
must be resolved before this can be a generally available upstream integration.

## Source updates

Run the source preparation module with an available Bridge Python 3.11 or 3.12
and Node 24+. Gym itself uses its normal Python 3.13 environment.

```bash
python -m responses_api_agents.ember_bridge_agent.source \
  --repository "$BRIDGE_REPOSITORY" --ref main \
  --cache-dir /scratch/ember-bridge-cache \
  --python /opt/python3.11/bin/python --node /opt/node24/bin/node
```

`repository` is a clone URL without a token. Use Git's credential helper for private
access. `component_dir` defaults to `groot/vlm/external_deps`. The preparer fetches
the selected ref, resolves it to one commit, creates an isolated checkout and
invokes Bridge's own `scripts/prepare.py`. That builder produces a matching
Nanoverse Python wheel, compiled npm package and standalone Bridge bundle. Gym
installs the bundle without patching project source. Preparation needs Git, Git LFS,
package indexes and the builder's pinned source dependencies; run it on a CPU
build node, not a GPU allocation or a shared login node.

The cache must live outside every Git checkout. Failed builds remain with their
`build.log` for inspection. A completed cache is reused after checking its artifact
and installed source hashes. Treat this cache as operator-owned installation state: do not edit it.

A server resolves a floating branch **once at startup**, then holds that commit
for its lifetime. Restarting with `main` picks up a new revision. For repeatable
evaluation set `source.ref` to a full commit hash; a cached commit works offline.
Every result records the selected commit, Nanoverse revision, tool versions and
release hashes. Updates never change an episode in progress.

The source registry controls supported harness/profile pairs. This workload uses
OpenCode 1.18.32 with the `teacher` profile and reasoning `xhigh`; the selected
source and model endpoint must support that pair. The SDK 1.4 preset uses RGB
observations and EEF control without auxiliary policy services.
Build and qualify the source owner's native
OpenCode runtime on CPU resources using its `docs/native-execution.md`,
`scripts/build-opencode-native.py`, `scripts/qualify-opencode-native.py` and
`scripts/prepare-native.py`. Set `harness_executable` to the emitted
`opencode/runtime` and `guest_programs` to that deployment's `programs.json`.
The stock npm executable alone cannot bind native TTC tools. Runtime versions,
digests and build receipts are checked by the source owner.

## Native runtime and shutdown

Use the prepared interpreter and wheel printed by the preparation command to run
the source owner's installer:

```bash
"$BRIDGE_PYTHON" -m nanoverse.install install libero \
  --root /scratch/libero --wheel "$NANOVERSE_WHEEL"
"$BRIDGE_PYTHON" -m nanoverse.install verify libero --root /scratch/libero
```

The LIBERO installer owns its separate Python 3.10 simulator environment and
assets. `native.python`, `native.root` and `native.libero_config` must refer to that
installation. Each episode checks the simulator interpreter's Nanoverse revision
against the prepared release. Updating Bridge may therefore require installing
its matching wheel into a freshly staged simulator runtime and verifying it.
The source's TTC manifest must name a working, integrity-checked TTC deployment
and guest archive, with the isolation and resource limits appropriate to the host.
Gym requires `sdk.execution: ttc` and does not create an isolation bypass.

Set `sdk.python` (the workload's `libero_sdk_python`) to a separate clean Python
interpreter, without Nanoverse, EmberBridge or simulator packages. The source
stages its public client there and refuses private-module imports before starting
the simulator. TTC then rebuilds the client wrappers for its guest interpreter.
The Bridge interpreter and the native simulator interpreter serve different roles
and must not replace this public client interpreter.

Source profiles using gated feedback also use the source-owned OpenAI Decisions API.
Set `decisions_key_file` to an operator-owned file with permissions 0600, containing
`OPENAI_DECISION_API_KEY` (or the source-supported alias `OPENAI_DECISION_API`).
The launcher reads that file through `OPENAI_DECISION_API_KEY_FILE`, removes the
credential variables and passes an inherited descriptor to its private Decisions
broker. The guest receives no provider credential. This preserves the source's
default gates and thresholds. These auxiliary requests use the source's own
provider/retry policy and retain their usage in native evidence; they do not pass
through Gym's actor model server or its model-call capture.

One server owns one episode at a time and requires `num_workers: 1`. Each episode
gets a loopback model endpoint and random capability; actual provider secrets
stay in Gym's stock model server. The proxy forwards each Responses POST once,
including streaming, with Gym's rollout correlation. The supported OpenCode
profile stops on provider failures without automatic inference retries. The
parent process receives a minimal environment.

The source's supervisor owns process cleanup. Gym checks its cleanup receipt plus
SDK and TTC shutdown evidence before reusing the worker. A persistent
`output_root/active.json` prevents reuse after a server crash or uncertain cleanup.
Inspect that episode and confirm all owned processes and TTC resources are gone
before manually removing the journal. A second server cannot share an active
output root. Give every deployment its own output root.
Keep that root on persistent storage so quarantine survives node loss. A batch
launcher using node-local episode files must separately refuse restarting an
uncertain attempt after scheduler requeue.

The Linux launcher also sends SIGTERM to that supervisor if its Gym worker dies,
so the source owner can reap descendants after abrupt parent loss.

A repeated `_ng_rollout_id` returns the persisted reply only when the complete
task, configuration and source identity match. An interrupted attempt is never
automatically replayed against the simulator.

Gym collector indices also derive the canonical rollout and attempt ID used for
journaling and model-call capture. A collector's run ID namespaces its journal,
so independent collections do not reuse each other's completed receipts.

Request rows select only `cell`,
`init_index` (0–49) and `seed`; they cannot replace the runtime or grading rules.
Rows must keep `responses_create_params.input` and `tools` empty. Actor token and
reasoning settings come from the Bridge configuration, not collector sampling flags.

## Results and validation

`ember_bridge.cli result` is the scoring authority. Verified binary `scored`
outcomes keep their native reward. Declared decision-budget, deadline, context
and no-progress policy failures count with reward zero. Provider, invalid,
missing and infrastructure outcomes set `mask_sample: true` and
`evaluation_completed: false`; their placeholder zero is not a measured failure.
Unconfirmed cleanup also masks the row. Keep these exclusions visible when
reporting success rate and completion rate.

`response` is the last model response actually observed by the proxy. If none
exists it is an empty, labelled control envelope with no fabricated message or
usage. When model-call capture is enabled, `ng_trajectory` projects the actual proxy
requests and responses into Gym turns, joins calls by the model server and
response ID, and reads completion only from the actor receipt. Missing exchanges
and tool execution evidence are explicit observation gaps. The proxy sees tool
requests and subsequent model inputs; it cannot certify tool execution timing
or the last tool result when no further model request follows. Keep the
source-owned episode evidence alongside Gym capture. Do not claim complete
tool observability, harness session conformance or RL token parity.

```bash
python -m unittest discover -s responses_api_agents/ember_bridge_agent/tests -v
```

The component tests cover source selection, proxy forwarding, grade admission,
cleanup and interrupted-attempt behavior. A real Gym/model/LIBERO rollout and
inspection of its evidence, completion masks and quality summary are still
required before marking this workload verified or proposing it for merge.
