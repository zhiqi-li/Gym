# LIBERO SDK smoke workload

This experimental environment runs five initial states of
`libero_goal/put_the_bowl_on_the_plate` through the
[EmberBridge agent](../../responses_api_agents/ember_bridge_agent/README.md).
The source owns the prompt and native binary verification. The committed rows
are task selectors, not a published LIBERO benchmark split. A smoke result must
not be reported as LIBERO suite accuracy.

Install Gym normally and prepare the source, native simulator and TTC deployment
as described in the agent README. LIBERO uses MuJoCo/EGL and can use a compatible
H100 installation; this workload does not need an RTX simulator. Availability of
EGL libraries, assets, TTC and the selected model still needs a real node check.

Put local settings in the root ignored `env.yaml`:

```yaml
libero_bridge_repository: <operator-accessible Git clone URL>
libero_bridge_ref: <commit hash, or main for updates on restart>
libero_bridge_cache: /scratch/ember-bridge-cache
libero_bridge_python: /opt/python3.11/bin/python
libero_bridge_node: /opt/node24/bin/node
libero_harness: opencode
libero_harness_executable: /scratch/native-deployment/opencode/runtime
libero_ttc_manifest: /scratch/ttc/deployment.json
libero_guest_programs: /scratch/native-deployment/programs.json
libero_decisions_key_file: /run/secrets/decisions.env
libero_output_root: /episodes
libero_native_python: /scratch/libero/venv/bin/python
libero_native_root: /scratch/libero/LIBERO-PRO
libero_native_config: /scratch/libero/libero-config
libero_sdk_python: /opt/sdk/bin/python3
policy_base_url: <Responses-compatible provider URL>
policy_api_key: <provider credential>
policy_model_name: <model identifier>
```

The preset uses SDK 1.4, the OpenCode `teacher` profile and reasoning `xhigh`,
with RGB observations, a 100-call budget and 1800-second episode deadline.
Use a model supported by that source profile. No VLA policy service is used.

Bind a persistent host evidence directory to a short container path such as
`/episodes`. OpenCode creates a Unix socket below each episode's output directory;
the complete socket path must fit 107 bytes. Gym appends a 64-character attempt
identifier, so long host paths should be shortened with a bind mount. A symlink
does not shorten the resolved output path. Keep the binding stable across restarts
so the attempt journal and cleanup quarantine survive node loss.

Start the composition on the same GPU node as its simulator:

```bash
capture_dir="$PWD/results/libero-smoke-model-calls"
gym env start --config environments/libero_sdk/config.yaml \
  ++observability_enabled=true ++model_call_capture_dir="$capture_dir"
gym eval run --no-serve --config environments/libero_sdk/config.yaml --agent libero_agent \
  --input environments/libero_sdk/data/example.jsonl \
  --output results/libero-smoke.jsonl --num-repeats 1 --concurrency 1 \
  ++observability_enabled=true ++model_call_capture_dir="$capture_dir"
```

Keep source preparation, package installation and dataset downloads on CPU
resources. The GPU allocation is for the evaluation process. Select an account
authorized for standalone evaluation; training-only accounts are unsuitable.
No scheduler or machine-specific settings are embedded in this environment.

The native installer may require network access and substantial disk space.
Source access and simulator asset licensing are operator prerequisites. The Gym
adapter is Apache-2.0; the selected EmberBridge release includes an MIT license,
but the paired Nanoverse distribution and simulator assets require a separate
license and access review. The manifest therefore keeps `licensing: unknown`.
This workload remains experimental and is not a generally available benchmark. Inspect `mask_sample`,
`evaluation_completed`, `outcome_class`, source provenance and cleanup evidence
alongside rewards. See the agent README for update and failure handling.
