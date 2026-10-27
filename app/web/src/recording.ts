import { api } from "./api";

/** Save the original take as soon as recording stops, independently of inference. */
export function recordingArchive(root: HTMLElement, mode: "real" | "demo") {
  let generation = 0;
  let url = "";
  function clear() {
    generation++;
    if (url) URL.revokeObjectURL(url);
    url = "";
    root.replaceChildren();
  }
  function save(wav: Blob) {
    clear();
    const current = generation;
    const status = document.createElement("p");
    status.className = "hint";
    status.setAttribute("role", "status");
    const download = document.createElement("a");
    download.className = "btn ghost";
    download.textContent = "Download WAV";
    url = URL.createObjectURL(wav);
    download.href = url;
    download.download = `recording_${new Date().toISOString().replace(/[:.]/g, "-")}.wav`;
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "btn ghost";
    retry.textContent = "Retry saving";
    retry.hidden = true;
    root.append(status, download, retry);
    async function upload() {
      retry.hidden = true;
      status.textContent = "Saving original WAV…";
      try {
        const saved = await api.saveRecording(wav, mode);
        if (current !== generation) return;
        download.download = saved.filename;
        status.textContent = `Saved: ${saved.filename}`;
      } catch (e: any) {
        if (current !== generation) return;
        status.textContent = `WAV not saved on server: ${e.message}. You can still download it.`;
        retry.hidden = false;
      }
    }
    retry.onclick = upload;
    void upload();
  }
  return { save, clear };
}
