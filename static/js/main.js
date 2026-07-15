// ---------- Tab switching ----------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

// ---------- Helper: capture a <video> frame as a base64 JPEG ----------
function captureFrame(videoEl) {
  const canvas = document.createElement("canvas");
  canvas.width = videoEl.videoWidth || 640;
  canvas.height = videoEl.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.8);
}

// ---------- Generic webcam controller ----------
function setupCameraLoop({ videoId, outputId, startBtnId, stopBtnId, endpoint, onResult }) {
  const video = document.getElementById(videoId);
  const output = document.getElementById(outputId);
  const startBtn = document.getElementById(startBtnId);
  const stopBtn = document.getElementById(stopBtnId);

  let stream = null;
  let intervalId = null;

  async function start() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      startBtn.disabled = true;
      stopBtn.disabled = false;

      intervalId = setInterval(async () => {
        if (video.videoWidth === 0) return;
        const frame = captureFrame(video);
        try {
          const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: frame }),
          });
          const data = await res.json();
          if (data.image) output.src = data.image;
          if (onResult) onResult(data);
        } catch (err) {
          console.error("Frame processing failed:", err);
        }
      }, 700);
    } catch (err) {
      alert("Could not access camera: " + err.message);
    }
  }

  function stop() {
    if (intervalId) clearInterval(intervalId);
    if (stream) stream.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }

  startBtn.addEventListener("click", start);
  stopBtn.addEventListener("click", stop);
}

// ---------- Tab 1: Live Detect ----------
setupCameraLoop({
  videoId: "detect-video",
  outputId: "detect-output",
  startBtnId: "detect-start",
  stopBtnId: "detect-stop",
  endpoint: "/api/detect",
  onResult: (data) => {
    document.getElementById("detect-count").textContent = `${data.count} face(s)`;
  },
});

// ---------- Tab 2: Live Recognize ----------
setupCameraLoop({
  videoId: "recognize-video",
  outputId: "recognize-output",
  startBtnId: "recognize-start",
  stopBtnId: "recognize-stop",
  endpoint: "/api/recognize",
  onResult: (data) => {
    const list = document.getElementById("recognize-results");
    list.innerHTML = "";
    (data.results || []).forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r.confidence != null ? `${r.name} (${r.confidence}%)` : r.name;
      list.appendChild(li);
    });
  },
});

// ---------- Tab 2: Register form ----------
document.getElementById("register-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = document.getElementById("register-name").value;
  const files = document.getElementById("register-images").files;
  const statusEl = document.getElementById("register-status");

  const formData = new FormData();
  formData.append("name", name);
  for (const file of files) formData.append("images", file);

  statusEl.textContent = "Uploading and training...";

  try {
    const res = await fetch("/api/register", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      statusEl.style.color = "var(--danger)";
      statusEl.textContent = data.error;
      return;
    }

    statusEl.style.color = "var(--green)";
    statusEl.textContent = `Saved ${data.saved} face(s) for "${data.name}". Model now knows ${data.people_count} people.`;

    const list = document.getElementById("people-list");
    const alreadyListed = [...list.children].some((li) => li.textContent === name);
    if (!alreadyListed) {
      list.querySelector(".empty")?.remove();
      const li = document.createElement("li");
      li.textContent = name;
      list.appendChild(li);
    }

    document.getElementById("register-form").reset();
  } catch (err) {
    statusEl.style.color = "var(--danger)";
    statusEl.textContent = "Something went wrong: " + err.message;
  }
});

// ---------- Tab 3: Upload & Detect ----------
document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("upload-image");
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  const res = await fetch("/api/upload", { method: "POST", body: formData });
  const data = await res.json();

  if (data.error) {
    alert(data.error);
    return;
  }

  document.getElementById("upload-output").src = data.image;

  const list = document.getElementById("upload-results");
  list.innerHTML = "";
  (data.results || []).forEach((r) => {
    const li = document.createElement("li");
    li.textContent = r.confidence != null ? `${r.name} (${r.confidence}%)` : r.name;
    list.appendChild(li);
  });
});
