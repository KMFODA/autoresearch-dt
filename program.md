# autoresearch-dt

This is an experiment to have the LLM do its own research on optimizing the convergence and throughput of a distributed training setup.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch-dt/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-dt/<tag>` from current master.
3. **Read the in-scope files**: The repo is relatively small. Read the files in these folders for full context:
   - `exogym/trainer.py` - the trainer script used to train your model.
   - `nanogpt/nanogpt.py` - the model you are training.
   - `strategy.py` — the file you modify. Communication and optimization strategies.
   - `evaluate.py` — how your distributed training strategy is evaluated.
4. **Verify data exists**: Check that `~/.cache/huggingface/hub` contains data shards and a tokenizer. If not, tell the human to run `uv run precache_dataset.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on multiple GPUs. The training script runs for a **fixed number of steps**. You launch it simply as: `uv run /root/DisTrOpZ/evaluator/evaluation_sandbox.py`.

**What you CAN do:**
- Modify `strategy.py` — this is the only file you edit. Everything is fair game: optimizer, communication strategy and communication frequency.

**What you CANNOT do:**
- Modify `evaluate.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (step budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `main` function in `evaluate.py` gives you the ground truth results.
- Modify any of the scripts in the exogym, nanogpt or strategies folders
- Use any collectives other than dist.all_reduce, dist.reduce, dist.broadcast, dist.all_gather, dist.scatter or dist.all_to_all

**The goal is simple: create a distributed training strategy that achievest the lowest loss, communication and throughput metrics.** Everything is fair game: changing the optimizer, the hyperparameters, the batch size, communication protocol or the communication frequency. The only constraint is that the code runs without crashing.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful loss or communication reductions or throughput gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 loss improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 loss improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
communication:    70,264,572,848
loss:             6.26
```

You can extract the key metric from the log file:

```
grep "^loss:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	communicaiton loss	status	description
```

1. git commit hash (short, 7 chars)
2. communication cost (e.g. 70,264,572,848) — use 0.000000 for crashes
3. loss achieved (e.g. 6.49) — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	communicaiton  loss	status	description
a1b2c3d	70,264,572,848 6.49	keep	   baseline
b2c3d4e	70,264,572,848 6.30	keep	   diloco 2bit error feedback
c3d4e5f	70,264,572,848 6.64	discard	Muloco 2Bit Ef
d4e5f6g	70,264,572,848 6.2	crash	   double model width (OOM)   
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-dt/mar5` or `autoresearch-dt/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `strategy.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run /root/autoresearch-dt/evaluate.py.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If communication or loss are improved (lower), you "advance" the branch, keeping the git commit
9. If both commuincation and loss are equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
