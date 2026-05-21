# Kimodo Building Motion Experiments

This context describes the language used for building-related motion generation experiments with KIMODO and G1-RP.

## Language

**Building Motion Generation Project**:
A project that uses KIMODO/G1-RP to generate reusable prior motions for building-related robot actions.
_Avoid_: Wall-brushing script project, single-task demo.

**G1-RP Motion Experiment**:
A building-action experiment that generates motions with the G1 robot model trained on the Rigplay dataset.
_Avoid_: Generic KIMODO experiment when the G1-RP model is specifically meant.

**Web-Equivalent Generation Path**:
A script generation flow whose qualitative result should match the motion behavior observed from the web UI for the same task setup.
_Avoid_: Bit-for-bit reproduction, browser automation.

**Legacy Wall-Brush Script**:
The older task-specific wall-brushing generation path that produced misleading behavior relative to the web UI.
_Avoid_: Keeping it as an active supported path for first-stage prior generation.

**Raw Base Motion**:
The motion produced by KIMODO before any downstream optimization for task contact, collision, or trajectory refinement.
_Avoid_: Final optimized motion, postprocessed motion.

**Postprocessed Prior Motion**:
A **Raw Base Motion** after lightweight cleanup that makes it more suitable as a **Building Motion Prior**.
_Avoid_: Final deployed controller, heavily re-authored motion.

**Raw-to-Postprocessed Lineage**:
The traceable relationship from a **Raw Base Motion** to any **Postprocessed Prior Motion** derived from it.
_Avoid_: Keeping only the postprocessed file, overwriting the raw output, losing the processing history.

**Building Motion Prior**:
A curated building-action motion that can seed later policy learning or RL training.
_Avoid_: Throwaway generated sample, final trained policy.

**DreamControl-Style Prior Training**:
An RL training approach where generated prior motions are used as reference trajectories for motion-tracking rewards while task-specific rewards still enforce task completion.
_Avoid_: Treating priors as behavior-cloning data only, requiring the deployed policy to replay a fixed reference trajectory.

**Deferred RL Export**:
The decision to preserve prior motion information needed for future RL training without implementing the RL dataset/export pipeline during the **First-Stage Loop**.
_Avoid_: Blocking prior collection on simulator integration or RL algorithm choices.

**Prior Admission Standard**:
The review bar for accepting a motion as a **Building Motion Prior**: correct action semantics, minimal extraneous movement, and enough structure for later postprocessing or RL to use.
_Avoid_: Constraint error alone, visual quality alone.

**Admission Blocker**:
A motion flaw that should prevent a **Raw Base Motion** from becoming a **Building Motion Prior**, such as a sudden start jump, obvious turn-around, looping, unrelated arm waving, wrong action semantics, excessive body drift, or unrelated motion after the task is complete.
_Avoid_: Small wall offset, imperfect coverage, contact-detail inaccuracy.

**Refinement Debt**:
An acceptable remaining issue in a semantically clean **Raw Base Motion** that can be handled later by postprocessing, optimization, or RL, such as a small fixed end-effector offset from the wall, imperfect line tracking, incomplete stroke coverage, or missing contact-force realism.
_Avoid_: Treating strange extra motions as merely refinable.

**Motion Prior Collection**:
A set of curated **Building Motion Priors** covering multiple building-related actions.
_Avoid_: Single experiment output folder, unreviewed generation batch.

**Candidate Prior Set**:
A small reviewable set of **Raw Base Motions** for an **Anchor Task**. It preserves useful variation before one or more motions are promoted to **Building Motion Priors**.
_Avoid_: Final fixed benchmark set, one-off unreviewed batch, permanent requirement for every future task.

**Named Candidate**:
A motion candidate with a stable human-readable identifier inside an **Executable Task Spec** and output folder.
_Avoid_: Relying only on array index, unnamed gallery motion, ambiguous output file.

**Prior Recipe**:
The reproducible recipe for a motion candidate, especially its prompt, constraint points, model choice, generation settings, and any postprocessing treatment.
_Avoid_: Untracked manual generation, motion file without its prompt or constraints, postprocessing result without its processing recipe.

**Recorded Seed**:
The actual random seed used for a generated motion candidate, stored in its **Prior Recipe** and manifest.
_Avoid_: Unrecorded randomness, relying on run logs to recover seed, changing seed without candidate identity.

**Seed Robustness Check**:
A small multi-seed check for an important prompt or recipe to see whether useful motion quality is reproducible.
_Avoid_: Expanding every prompt into a large seed sweep during the first stage.

**Prompt Strategy**:
The prompting approach used to generate a motion candidate, such as a single whole-task prompt or a sequence of simpler phase prompts.
_Avoid_: Treating prompt text as an incidental note, assuming one prompting style is always best.

**Prompt Search Candidate**:
A candidate **Prompt Strategy** or prompt variant being compared for an **Anchor Task**.
_Avoid_: Untracked prompt tweaking, declaring one strategy optimal before review.

**Small Prompt Search**:
A reviewable prompt search approach that compares a small hand-designed set of **Prompt Search Candidates** rather than automatically generating a large prompt batch.
_Avoid_: Large unreviewable prompt sweeps, hidden automatic prompt mutation.

**Composed Action Prompt**:
A **Prompt Strategy** that describes the target behavior as connected simple action phases, such as approach, brush one line, and stop naturally.
_Avoid_: Assuming composition is the only valid prompt search method.

**Native Multi-Prompt Generation**:
A KIMODO generation mode that uses multiple prompt segments in one generation call to produce a single continuous motion.
_Avoid_: Offline stitching of separate generated motion files, treating prompt segments as postprocessing.

**Visible Prompt Segments**:
Prompt segments whose labels, text, and time ranges are preserved in the recipe and displayed in the review interface, regardless of whether the boundaries were hand-authored or suggested by AI/tooling.
_Avoid_: Hidden segment boundaries, segment text visible only in code, forcing users to hand-author every frame range.

**Segment Source**:
The recorded origin of prompt segment boundaries, such as hand-authored, auto-generated, or AI-suggested.
_Avoid_: Silent default segmentation, losing whether a timing choice was authored or inferred.

**Task Spec**:
A reusable description of one building-action generation task. It captures the action prompt, constraint points, generation settings, model choice, and expected outputs so the shared generation path can produce motion candidates.
_Avoid_: One-off hardcoded script, undocumented command-line invocation, task hidden inside code.

**Executable Task Spec**:
A JSON **Task Spec** that can be passed directly to the generation script to produce motions without translating the recipe into a separate custom script.
_Avoid_: Spec that only documents intent, hand-copying values into code, script-only configuration.

**Task Spec Validation**:
Schema or strict validation for an **Executable Task Spec** before generation starts.
_Avoid_: Best-effort parsing that silently drops prompt, constraints, seed, model, or candidate identity.

**Declared Constraint Point**:
A task constraint that explicitly records end effector, frame or time, position, coordinate frame, generation usage, review visibility, postprocessing usage, and task label.
_Avoid_: Unlabeled coordinate triplet, hidden coordinate system, constraint point visible only in code, reference point that is not passed to generation when the task requires constraint following.

**Postprocessing Treatment**:
The explicit cleanup or refinement method applied after raw generation, tracked as part of a **Prior Recipe** when it is used.
_Avoid_: Silent edits, overwriting the raw motion, treating postprocessing as part of the model's raw output.

**Executable Postprocessing Treatment**:
A JSON-declared **Postprocessing Treatment** that can be run by the pipeline with explicit parameters and lineage back to the source raw motion.
_Avoid_: Hardcoded cleanup, undocumented manual smoothing, postprocessing chosen only from chat history.

**Postprocessing Candidate**:
A candidate cleanup method considered for improving **Raw Base Motions** while preserving their useful action semantics.
_Avoid_: Declaring one treatment as permanent default before comparison, treating postprocessing as a way to fix admission blockers.

**Postprocessing Admission Standard**:
The review bar for accepting a **Postprocessing Candidate**: it should reduce **Refinement Debt** such as contact or smoothness issues without creating new **Admission Blockers** or changing the action semantics.
_Avoid_: Optimizing geometry metrics while making the motion unnatural.

**Prior Run Folder**:
The standardized output directory for one generation run, containing the executable task spec, raw motions, postprocessed derivatives, metrics, review records, gallery, and manifest.
_Avoid_: Scattered logs, ad hoc output names, motion files separated from their recipes.

**Remote Experiment Artifact**:
A generated motion artifact kept on the remote server, such as run folders, raw motions, postprocessed motions, galleries, metrics, and review files.
_Avoid_: Committing large generated motion outputs to the source repository by default.

**Local Review Artifact**:
A lightweight artifact synced locally for responsive review, such as motion `.npz` files, manifests, review files, metrics, and galleries.
_Avoid_: Requiring remote streaming for every visual inspection, downloading large unnecessary media by default.

**Local Review Folder**:
A locally synced copy of a **Prior Run Folder** used as the primary input for the **Motion Review Interface**.
_Avoid_: Reading motions directly from the remote server during normal visual review.

**Repo-Persisted Prior Recipe**:
A validated or important prior recipe and review conclusion preserved in the source repository so it survives remote server loss.
_Avoid_: Keeping all important findings only on the remote server, committing large motion artifacts by default.

**Executable Recipe File**:
A repository-stored JSON task spec that can be passed directly to the generation script to reproduce a validated or important prior recipe.
_Avoid_: Markdown-only recipe, values that must be manually copied into code.

**Recipe Note**:
A repository-stored human-readable explanation of a prior recipe, including why it was kept, review observations, admission blockers avoided, and remaining refinement debt.
_Avoid_: Machine-only JSON with no rationale, chat-only review history.

**Recipe Note Draft**:
A generated Markdown draft created from a candidate's recipe, manifest, metrics, artifacts, and review record for later human editing before repo persistence.
_Avoid_: Requiring users to manually reconstruct accepted-candidate context from logs.

**Artifact Reference**:
A lightweight reference from repo documentation to a remote run folder, gallery, candidate name, screenshot, or other review artifact without committing large generated media by default.
_Avoid_: Large motion/video files in the repo, recipe notes with no link back to reviewed artifacts.

**Prior Manifest**:
The run-level index that links each **Named Candidate** to its **Prior Recipe**, raw motion path, postprocessed derivatives, metrics, review record, and visualization entry.
_Avoid_: Discovering outputs by manual folder inspection, hidden provenance, gallery-only indexing.

**Motion Review Interface**:
The visualization and inspection surface used to compare generated motion candidates, constraints, metrics, recipes, and raw-to-postprocessed lineage.
_Avoid_: Treating visualization as only a playback window, hiding constraint points or candidate identity.

**Review-First Interface**:
A first-stage **Motion Review Interface** focused on selecting and diagnosing motion candidates before building full editing tools.
_Avoid_: Full motion editor, constraint authoring workspace, postprocessing parameter studio.

**First-Stage Loop**:
The initial complete workflow for the project: generate a **Candidate Prior Set** for **One-Row Wall Brushing**, preserve each candidate's **Prior Recipe**, inspect metrics and visualization, record human review, and keep raw-to-postprocessed lineage when postprocessing is used.
_Avoid_: Full building-action library, RL training integration, full motion editor.

**One-Command Prior Run**:
A command that consumes an **Executable Task Spec** and produces a complete **Prior Run Folder**, including generated motions, recipes, metrics, manifest, and review gallery.
_Avoid_: Manually chaining several scripts before a candidate can be reviewed.

**Remote Prior Run**:
A **One-Command Prior Run** executed on the remote server where the KIMODO model, cache, and GPU environment are available.
_Avoid_: Assuming local generation works without the model/GPU environment.

**Candidate Review Record**:
The structured human review attached to a motion candidate, including review status, admission blockers, refinement debt, and notes explaining why the candidate was accepted, rejected, or deferred.
_Avoid_: Chat-only judgment, unlabeled gallery output, metric-only decision.

**Review Status**:
The structured status assigned to a candidate during review, such as raw accepted, postprocessed accepted, needs postprocess, needs regeneration, or rejected.
_Avoid_: A single ambiguous "accepted" label that hides whether the raw or postprocessed motion is trusted.

**Run-Level Review File**:
A first-stage review file that indexes **Candidate Review Records** by **Named Candidate** inside a **Prior Run Folder**.
_Avoid_: Scattered review notes, review status embedded only in HTML, requiring one review file per candidate before the collection grows.

**File-Based Review State**:
Review state stored in local JSON files, especially the **Run-Level Review File**, rather than a database or hidden UI state.
_Avoid_: Review decisions that exist only inside the interface session.

**Diagnostic Metrics**:
Quantitative signals used to inspect and rank motion candidates, such as start jump, root drift, body rotation, end-effector trajectory error, constraint error, and extra motion after task completion.
_Avoid_: Treating metrics as the final admission decision.

**Anchor Task**:
A first task used to validate the generation, review, and curation workflow before expanding to more building-action categories.
_Avoid_: Entire project scope, final benchmark suite.

**One-Row Wall Brushing**:
The first wall-brushing milestone: approach the wall, perform one clear horizontal brushing stroke, then return or stop naturally.
_Avoid_: Full-wall coverage, multi-row painting.

**Motion Quality First**:
The admission priority that semantic cleanliness and lack of extraneous movement matter more than small end-effector trajectory errors.
_Avoid_: Constraint-only ranking, metric-only acceptance.

**Semantic Task Success**:
The early-stage success standard for a building-action prior: the motion clearly expresses the intended action with natural body behavior and minimal extraneous movement, even if precise contact or geometry matching remains unfinished.
_Avoid_: Treating physical contact accuracy as the only success criterion.

**Contact Precision Debt**:
A form of **Refinement Debt** where the motion semantics are useful but the end effector does not yet exactly match the wall surface, tool path, or contact behavior.
_Avoid_: Rejecting a semantically clean motion only because wall contact is not exact.

**Canonical Start Motion**:
The stable G1 starting state used for prior generation. It should match the posture of the real G1 robot when it enters debug mode, so generated motions do not begin with a discontinuous jump or sudden correction.
_Avoid_: Constraint-calibration pose only, arbitrary generated first frame, visually convenient pose that does not match G1 debug mode.

## Relationships

- The **Building Motion Generation Project** produces a **Motion Prior Collection**.
- Wall brushing is the first **Anchor Task** for the **Building Motion Generation Project**.
- **One-Row Wall Brushing** is the first milestone of the wall-brushing **Anchor Task**.
- **One-Row Wall Brushing** follows **Motion Quality First** during early prior selection.
- Early **One-Row Wall Brushing** is judged by **Semantic Task Success** before exact wall contact.
- Early **One-Row Wall Brushing** should produce a **Candidate Prior Set** rather than a single locked-in motion.
- A **Candidate Prior Set** may be generated from one **Executable Task Spec** containing multiple **Named Candidates**.
- Every motion in a **Candidate Prior Set** should keep its **Prior Recipe**.
- Every generated candidate should keep its **Recorded Seed**.
- Multi-seed generation should be limited to **Seed Robustness Checks** for selected promising recipes during the **First-Stage Loop**.
- Prompt search may compare multiple **Prompt Search Candidates**, including **Composed Action Prompts**.
- First-stage prompt search should be **Small Prompt Search** so candidates remain reviewable.
- **Composed Action Prompts** should use **Native Multi-Prompt Generation** rather than offline motion stitching.
- Prompt segment labels, text, and time ranges should be **Visible Prompt Segments** in the review interface.
- When segment boundaries are inferred, their **Segment Source** should be recorded in the recipe or manifest.
- For first-stage wall brushing, constraint points in a **Prior Recipe** are generation constraints by default: they should be passed to KIMODO, visible in the **Motion Review Interface**, and optionally reused by postprocessing.
- Building-action tasks should be represented as **Task Specs** that use the shared **Web-Equivalent Generation Path**.
- First-stage **Task Specs** should be **Executable Task Specs**.
- **Executable Task Specs** require **Task Spec Validation** before generation.
- First-stage generation is fixed to the G1-RP model.
- Postprocessing should be represented as **Executable Postprocessing Treatments** when used.
- Postprocessing is disabled by default during the **First-Stage Loop** and should run only when explicitly requested by the **Executable Task Spec**.
- Existing good postprocessing methods may be evaluated as **Postprocessing Candidates**, but are not first-stage defaults.
- A **Postprocessing Candidate** must satisfy the **Postprocessing Admission Standard** before it is trusted for prior generation.
- Each generation run should write a standardized **Prior Run Folder** with a **Prior Manifest**.
- The **First-Stage Loop** should support a **One-Command Prior Run**.
- First-stage generation should run as a **Remote Prior Run** by default.
- Generated run outputs are **Remote Experiment Artifacts** by default and should stay on the remote server rather than being committed to the repo.
- Lightweight review outputs, including `.npz` motion files, may be synced as **Local Review Artifacts** for low-latency inspection.
- The **Motion Review Interface** should primarily load a **Local Review Folder**.
- Validated recipes and important review conclusions should become **Repo-Persisted Prior Recipes** so they survive remote server loss.
- A **Repo-Persisted Prior Recipe** should include both an **Executable Recipe File** and a **Recipe Note**.
- Accepted or promising candidates may generate a **Recipe Note Draft** from their review data.
- A **Recipe Note** should include **Artifact References** for the reviewed run when available.
- A **Motion Review Interface** should make candidate comparison and provenance visible during prior selection.
- The first **Motion Review Interface** should be a **Review-First Interface** for candidate comparison and diagnosis, not a full editing environment.
- The immediate project milestone is the **First-Stage Loop**, not the full RL training system.
- Each reviewed candidate should keep a **Candidate Review Record** alongside its **Prior Recipe** and metrics.
- A **Candidate Review Record** should use explicit **Review Status** values that distinguish raw and postprocessed acceptance.
- First-stage review records may be stored together in a **Run-Level Review File**.
- First-stage review decisions should use **File-Based Review State**.
- **Diagnostic Metrics** support review, but the **Prior Admission Standard** remains the final admission authority.
- A **Building Motion Prior** should begin from the **Canonical Start Motion** when the expected deployment setup uses that stable start.
- A **G1-RP Motion Experiment** produces one or more **Raw Base Motions**.
- A **Web-Equivalent Generation Path** is acceptable when the generated motion quality matches the web UI for the same task, even if the output is not numerically identical.
- The **Legacy Wall-Brush Script** should be removed from the supported first-stage generation path.
- A **Raw Base Motion** may become a **Building Motion Prior** after review and optional postprocessing.
- A **Postprocessed Prior Motion** remains a prior motion artifact, not a final policy.
- A **Postprocessed Prior Motion** must keep its **Raw-to-Postprocessed Lineage** and must not replace the original **Raw Base Motion**.
- A **Building Motion Prior** must satisfy the **Prior Admission Standard**.
- A **Building Motion Prior** is intended for **DreamControl-Style Prior Training** unless a later RL design changes this decision.
- RL export is a **Deferred RL Export** during the **First-Stage Loop**.
- A **Raw Base Motion** with an **Admission Blocker** should not be promoted to a **Building Motion Prior**.
- **Refinement Debt** is acceptable only when the motion still satisfies **Motion Quality First**.
- A **Motion Prior Collection** may later be used by RL training.

## Example Dialogue

> **Dev:** "Should the script reproduce every floating-point value from the web UI?"
> **Domain expert:** "No. For a **Web-Equivalent Generation Path**, it is enough that the **Raw Base Motion** has the same useful behavior for the **G1-RP Motion Experiment**."

> **Dev:** "Should the old wall-brush-specific script remain available as a supported path?"
> **Domain expert:** "No. Remove the **Legacy Wall-Brush Script** from the supported first-stage path to avoid misleading experiments."

> **Dev:** "Is wall brushing the whole project?"
> **Domain expert:** "No. Wall brushing is one validation task inside the **Building Motion Generation Project**; the goal is a **Motion Prior Collection** for later RL training."

> **Dev:** "Should we branch into many construction tasks immediately?"
> **Domain expert:** "Not before the first **Anchor Task** is useful. Wall brushing should validate the workflow, then we expand the **Motion Prior Collection**."

> **Dev:** "Should the first wall-brushing milestone keep only one best motion?"
> **Domain expert:** "No. Keep a **Candidate Prior Set** first, then promote the useful motions after review."

> **Dev:** "Can candidates be identified only by their position in a JSON array?"
> **Domain expert:** "No. Each candidate should be a **Named Candidate** so visual review and provenance remain clear."

> **Dev:** "If a candidate looks good, do we need to keep the prompt and constraint points?"
> **Domain expert:** "Yes. A candidate without its **Prior Recipe** is not suitable for a reusable prior collection."

> **Dev:** "Can the generation script choose a random seed without saving it?"
> **Domain expert:** "No. Every candidate needs a **Recorded Seed**."

> **Dev:** "Should every prompt variant run many seeds by default?"
> **Domain expert:** "No. Use **Seed Robustness Checks** only for selected promising recipes in the first stage."

> **Dev:** "Is prompt search just changing one whole-task sentence?"
> **Domain expert:** "No. It can compare **Prompt Search Candidates**, including **Composed Action Prompts** made from simple action phases."

> **Dev:** "Should prompt search automatically generate a large batch?"
> **Domain expert:** "No. The first stage should use **Small Prompt Search** with a small hand-designed set of variants."

> **Dev:** "Does composed prompting mean generating separate clips and stitching them?"
> **Domain expert:** "No. Use **Native Multi-Prompt Generation** so KIMODO produces one continuous motion from multiple prompt segments."

> **Dev:** "Does the user need to manually specify every segment frame?"
> **Domain expert:** "No. Segment boundaries may be suggested by AI/tooling, but they must become **Visible Prompt Segments** for review."

> **Dev:** "Can default segment timing be applied silently?"
> **Domain expert:** "No. If timing is inferred, record the **Segment Source**."

> **Dev:** "Can constraint points be stored as unlabeled XYZ values?"
> **Domain expert:** "No. They should be **Declared Constraint Points** and visible during review."

> **Dev:** "Can wall-brushing constraint points be only review references?"
> **Domain expert:** "No. In the **First-Stage Loop**, they are generation constraints by default and must be passed to KIMODO."

> **Dev:** "Should each new building action get its own custom generation script?"
> **Domain expert:** "No. Define a **Task Spec** and run it through the shared **Web-Equivalent Generation Path**."

> **Dev:** "Can the task spec be just a note describing what to generate?"
> **Domain expert:** "No. It should be an **Executable Task Spec** that the script can consume directly."

> **Dev:** "Can the generator accept incomplete task specs and guess missing fields?"
> **Domain expert:** "No. Use **Task Spec Validation** so missing critical fields fail before generation."

> **Dev:** "Can first-stage prior collection mix different robot models?"
> **Domain expert:** "No. The **First-Stage Loop** is fixed to G1-RP so generated priors remain compatible with the intended robot and RL setup."

> **Dev:** "Can postprocessing be hidden inside the script?"
> **Domain expert:** "No. When postprocessing is used, it should be an **Executable Postprocessing Treatment** with explicit parameters and lineage."

> **Dev:** "Should first-stage generation run postprocessing by default?"
> **Domain expert:** "No. Postprocessing is off by default and must be explicitly requested by the **Executable Task Spec**."

> **Dev:** "Can a postprocess method be accepted because it lowers wall error even if the motion becomes strange?"
> **Domain expert:** "No. It must satisfy the **Postprocessing Admission Standard**: reduce refinement debt without introducing admission blockers."

> **Dev:** "Can generated motions live in arbitrary log folders?"
> **Domain expert:** "No. Each run should produce a **Prior Run Folder** with a **Prior Manifest** for indexing."

> **Dev:** "Should generated motion outputs be committed to the source repo?"
> **Domain expert:** "No. Treat them as **Remote Experiment Artifacts** unless a later fixture or release decision says otherwise."

> **Dev:** "Should local review require streaming every motion from the remote server?"
> **Domain expert:** "No. Sync lightweight **Local Review Artifacts**, including `.npz` motions, for responsive inspection."

> **Dev:** "Should the review interface read remote paths during normal inspection?"
> **Domain expert:** "No. It should primarily load a **Local Review Folder**."

> **Dev:** "If the remote server disappears, should we lose the successful recipe?"
> **Domain expert:** "No. Preserve validated recipes and important review conclusions as **Repo-Persisted Prior Recipes**."

> **Dev:** "Is a Markdown note enough to preserve a successful prior recipe?"
> **Domain expert:** "No. Keep both an **Executable Recipe File** and a **Recipe Note**."

> **Dev:** "Should accepted candidates require manually writing recipe notes from scratch?"
> **Domain expert:** "No. Generate a **Recipe Note Draft** from the candidate's recipe, metrics, artifacts, and review record."

> **Dev:** "Should recipe notes commit large videos or motion outputs?"
> **Domain expert:** "No. Use **Artifact References** unless a later fixture/release decision says otherwise."

> **Dev:** "Is visualization only for playing back a generated motion?"
> **Domain expert:** "No. The **Motion Review Interface** should support candidate comparison, constraint inspection, and provenance review."

> **Dev:** "Should the first visualization pass include motion editing and constraint dragging?"
> **Domain expert:** "No. Start with a **Review-First Interface**: compare candidates, inspect constraints and recipes, and label issues."

> **Dev:** "Does the first-stage work need to include RL training?"
> **Domain expert:** "No. The **First-Stage Loop** ends when useful wall-brushing candidates are generated, inspected, reviewed, and preserved."

> **Dev:** "Should users manually chain generation, metrics, gallery, and manifest scripts?"
> **Domain expert:** "No. The first stage should provide a **One-Command Prior Run**."

> **Dev:** "Should first-stage generation run locally by default?"
> **Domain expert:** "No. Treat generation as a **Remote Prior Run** because the model and GPU environment live on the remote server."

> **Dev:** "Can human review live only in the conversation?"
> **Domain expert:** "No. Keep a **Candidate Review Record** so accepted and rejected priors remain explainable."

> **Dev:** "Can review status simply say accepted?"
> **Domain expert:** "No. **Review Status** should distinguish raw accepted from postprocessed accepted."

> **Dev:** "Does each candidate need its own review file immediately?"
> **Domain expert:** "No. In the first stage, a **Run-Level Review File** indexed by candidate name is enough."

> **Dev:** "Can review decisions live only in the browser UI?"
> **Domain expert:** "No. Use **File-Based Review State** so review survives refreshes and can be reused."

> **Dev:** "Can the lowest constraint error automatically become the accepted prior?"
> **Domain expert:** "No. **Diagnostic Metrics** help review, but the **Prior Admission Standard** decides."

> **Dev:** "For the first wall-brushing milestone, should the robot cover a whole wall?"
> **Domain expert:** "No. **One-Row Wall Brushing** is enough: approach, one horizontal stroke, and a natural stop."

> **Dev:** "This motion follows the target line more precisely but adds a strange turn before brushing. Should we pick it?"
> **Domain expert:** "No. Under **Motion Quality First**, a cleaner motion with a small trajectory offset is preferred."

> **Dev:** "Is the web UI start only used to calibrate constraints?"
> **Domain expert:** "No. The **Canonical Start Motion** should match the real G1 debug-mode posture so generated priors start without a sudden movement."

> **Dev:** "This sample misses the exact wall constraint by a few centimeters. Should we reject it?"
> **Domain expert:** "Not automatically. If it satisfies the **Prior Admission Standard**, it can still become a **Building Motion Prior** because postprocessing or RL may handle the remaining precision."

> **Dev:** "The brushing motion is clean but the hand does not exactly touch the wall. Is that a failure?"
> **Domain expert:** "Not at this stage. That is **Contact Precision Debt** if the motion has **Semantic Task Success**."

> **Dev:** "This motion brushes the wall but spins around once before starting. Can RL fix that later?"
> **Domain expert:** "No. That is an **Admission Blocker**, not **Refinement Debt**."

> **Dev:** "Can postprocessing force every hand frame exactly onto the wall?"
> **Domain expert:** "Only if the motion remains natural. A **Postprocessed Prior Motion** should clean up a usable prior, not rewrite it into an unnatural metric artifact."

> **Dev:** "If the postprocessed version is better, can we delete the raw motion?"
> **Domain expert:** "No. Keep the **Raw-to-Postprocessed Lineage** so we know what KIMODO produced and what cleanup changed."

> **Dev:** "Are building priors just behavior-cloning demonstrations?"
> **Domain expert:** "No. The current target is **DreamControl-Style Prior Training**: use priors as reference trajectories for RL tracking rewards while task rewards enforce success."

> **Dev:** "Does the first stage need to implement RL dataset export?"
> **Domain expert:** "No. Treat RL export as **Deferred RL Export** while preserving enough lineage and motion data for later."

## Flagged Ambiguities

- "Same as web" means qualitatively equivalent generation behavior, not exact numerical identity.
- G1 post-processing is not a decision variable for the current experiments; the current scope is **Raw Base Motion** generation with G1-RP.
- "Prior motion" means a curated **Building Motion Prior**, not necessarily an untouched KIMODO sample.
- A motion file without its **Prior Recipe** should not be treated as a reusable prior.
- Constraint accuracy is not the sole admission criterion for a **Building Motion Prior**.
- **Admission Blockers** and **Refinement Debt** should be separated during review.
- A **Postprocessed Prior Motion** should not overwrite or obscure its source **Raw Base Motion**.
- Exact wall contact is **Contact Precision Debt** during early wall-brushing prior generation, not the primary success criterion.
- The current RL target is **DreamControl-Style Prior Training**, but the exact RL algorithm and simulator integration remain future design decisions.
- Postprocessing is allowed as lightweight cleanup before RL, but it should not replace the need for a semantically useful **Raw Base Motion**.
- For early **One-Row Wall Brushing**, small trajectory offsets are acceptable when the motion is semantically clean.
- "Start pose" was too narrow; the intended concept is **Canonical Start Motion**, a deployment-aligned G1 debug-mode starting state.
- **Canonical Start Motion** names the desired deployment-aligned starting state; it does not by itself require adding a generation-time start constraint to every experiment.
