# All Improvements Implementation Plan

Goal: execute the approved nine-idea hardware plan on A/B/D/F through frozen evaluation and reporting.
Architecture: existing method registry, datasets, training state, campaign snapshots, sweep scheduler and persistent user services. No changes to raw data, committed splits or historical snapshots.
Spec: docs/updates/all-improvements-hardware-plan-2026-10-25.md.

- [x] Task1: implement LLRD and patient cap in optim/modeling/dataset; focused LR, resume and sampling tests.
- [x] Task2: implement classification pooling and class-aware PAFA in encoder/methods; reference, gradient and default-compatibility tests.
- [x] Task3: implement recording bags/head and waveform device augmentation/consistency in separate method modules; patient isolation, physical augmentation and gradient tests.
- [x] Task4: integrate parser/registry/trainer and genuine recording inference. Save per-epoch source OOF and source-only recording validation. Verify old defaults, resume, wrong-method rejection and no official-test CV access.
- [x] Task5: audit source preprocessing with fixed source-only checkpoints, 5/8s and one-factor resampling/fading contrasts. Preserve registered KAUH8s.
- [x] Task6: real GPU pilots; immutable per-release source snapshot; use existing sweep with persistent services. F <64GB project use and disk floor30GiB; archive to B before removing newly generated redundant files.
- [x] Task7: source3fold seed0 screening of all8 new arms; exact-anchor verification, up to3seed1 finalists, one2seed stack. Frozen gates, paired epochs, no target-based selection.
- [x] Task8: at most2new recipes, five-seed official+4LODO frozen evaluation, screening/KAUH exports, hash verification, paired report. Deadline admission26Oct23:00+07; report pending instead of silently extending reservation.

Execution ledger:
- 2026-10-26 01:45+07: A/B/D/F reachable, all GPUs idle. Previous50final fits and report complete. Parallel feature owners assigned disjoint files. Current source is dirty; preserve unrelated changes and use immutable filesystem campaign snapshots for training isolation, no automatic commits.

- 2026-10-26 02:08+07: Tasks1–4 implemented and integrated.159 focused/adjacent tests pass.8/8 real source GPU pilots pass; recording AMP backoff confirmed finite atscale8192. Four full-source epoch smoke fits A/B/D/F pass, persist cycle+recording OOF/checkpoints and no official-test predictions. CV official-test construction regression reproduced RED and fixed. Pooling embedding export contract fixed RED→GREEN; worker override and CVseedreservation fixed RED→GREEN. Immutable release build/campaigns/improvements-20261026 (200 hashed sources/configs) deployedallhosts; 27screen fits queued A7/B7/D6/F7. Persistent rootcontroller and perphaseworkers automatically select/confirm/stack/final/report; sourceaudit nativeCPUservices await certifiedanchor checkpoints. Originaldeadline26Oct23+07 retained.

- 2026-10-26 09:38+07: Complete:30screen+6confirm+25finalunits;3sourceaudits;75finalinputreceipts andallunitcheckpoint/outputhashesverified. SourceLLRD+0.553passed; otherarmsfailedsourcegate. No stackbecauseonesurvivor. FrozenfinalLLRDofficial62.313 versusgain6 62.416;LODOmacro60.610 versus61.093;noestablishedgain. Retainpreviouspipeline;nodeploymentchange. CPUauditimportbugfixedviaPYTHONPATH; optionalCUDAFP32auditpassedsourceparity(<4.1e-9 probabilitydifference) andall3foldauditsfinished34–39s. See docs/reports/all-improvements-2026-10-26.md and outputs/evaluation/improvements-20261026/.
