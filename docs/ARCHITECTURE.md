# Architecture

Diagrams of the final system. Sources: `auscult_trust/models/`, `auscult_trust/processing/correction.py`, `auscult_trust/evaluation/conformal.py`, `auscult_trust/training/methods/pafa.py`, `docs/decision-logs/2026-10-29-final-model-distilled.md`.

## 1. Inference pipeline

```mermaid
flowchart LR
    A["Recording (WAV)<br/>+ device name"] --> B{"Annotated<br/>cycles?"}
    B -- yes --> C["Breathing cycles<br/>cut / padded to 5 s"]
    B -- no --> D["Sliding windows<br/>8 s, step 4 s"]
    C --> E["Spectrum correction<br/>(figure 3)"]
    D --> E
    E --> F["Kaldi fbank<br/>128 bins, 25 / 10 ms"]
    F --> G["BEATs + PAFA student<br/>(figure 2)"]
    G --> H["Softmax, 4 classes<br/>normal / crackle / wheeze / both"]
    H --> I["Split-conformal set<br/>LAC, alpha 0.1"]
    H --> J["p(abnormal) = 1 - p(normal)<br/>mean over 8 s windows"]
    J --> K["Triage (2-class conformal)<br/>no referral / refer / uncertain / re-record"]
```

Cycle-level output (4 classes and sets) and recording-level screening (triage) share the same softmax.

## 2. Model: BEATs encoder with PAFA

```mermaid
flowchart TB
    X["fbank 798 x 128<br/>(8 s)"] --> P["Patch embedding<br/>16 x 16 patches"]
    P --> T["BEATs transformer encoder<br/>AudioSet-2M iter3+, fine-tuned"]
    T --> M["Mean over tokens<br/>768-d feature"]
    M --> CL["Linear classifier<br/>4 logits"]
    M --> PR["PAFA projection head<br/>(training only)"]
    CL --> L1["Classification loss<br/>CE, or soft-target CE when distilling"]
    PR --> L2["PCSL loss, weight 5<br/>patient cohesion / separation"]
    PR --> L3["GPAL loss, weight 0.005<br/>global patient alignment"]
```

At inference only the encoder and the linear classifier run. PAFA losses use patient identity and exist only at training time.

## 3. Spectrum correction (SC)

```mermaid
flowchart LR
    subgraph fit["Estimate (unlabelled audio only)"]
        S1["Clips of device d"] --> S2["Mean spectrum of d<br/>STFT 1024, hop 512, 16 kHz"]
        S3["Training devices"] --> S4["Reference =<br/>arithmetic mean of device spectra"]
        S2 --> S5["Gain per bin =<br/>reference / device spectrum"]
        S4 --> S5
        S5 --> S6["Bound the gain<br/>dynamic: +-20 dB every bin<br/>static: 50-2000 Hz free, no-op outside"]
    end
    subgraph apply["Apply"]
        W["Waveform"] --> ST["STFT"]
        ST --> MUL["x gain per bin"]
        S6 --> MUL
        MUL --> IS["inverse STFT"]
    end
```

Known device: coefficients saved with the model. Unknown device (microphone take): estimated from the recording's own segments. Training adds a random smooth per-bin gain (6 dB SD) in the final recipe.

## 4. Distillation: how AuscultTrust is trained

```mermaid
flowchart LR
    subgraph teachers["Teachers G4 (same seed)"]
        T1["anchor gain6-cv66"]
        T2["pcsl5_control"]
        T3["class_aware_light"]
        T4["multiscale"]
    end
    T1 --> AV["Equal-weight mean softmax<br/>on training cycles"]
    T2 --> AV
    T3 --> AV
    T4 --> AV
    AV --> ST["Student kd_g4_k1<br/>BEATs + PAFA, static SC, gain 6 dB<br/>soft targets only, T = 1, epoch 66"]
    ST --> CP["+ split-conformal layer"]
    CP --> OUT["AuscultTrust"]
```

The teachers are used at training time only; serving runs the single student.
