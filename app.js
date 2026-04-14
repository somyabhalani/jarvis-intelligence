const timeNow = document.getElementById("timeNow");
const dayName = document.getElementById("dayName");
const monthDay = document.getElementById("monthDay");
const coreIndex = document.getElementById("coreIndex");
const globalBars = document.getElementById("globalBars");
const powerPct = document.getElementById("powerPct");
const tempVal = document.getElementById("tempVal");
const weatherLabel = document.getElementById("weatherLabel");
const weatherLocation = document.getElementById("weatherLocation");
const weatherForecast = document.getElementById("weatherForecast");
const globalNewsStatus = document.getElementById("globalNewsStatus");
const globalNewsList = document.getElementById("globalNewsList");
const marketStatus = document.getElementById("marketStatus");
const marketList = document.getElementById("marketList");
const resourceList = document.getElementById("resourceList");
const uptimeChip = document.getElementById("uptimeChip");
const commandForm = document.getElementById("commandForm");
const commandInput = document.getElementById("commandInput");
// ...existing code...
const commandLog = document.getElementById("commandLog");
const statusDot = document.getElementById("statusDot");
const modeTypeBtn = document.getElementById("modeTypeBtn");
const modeListenBtn = document.getElementById("modeListenBtn");
const listenBtn = document.getElementById("listenBtn");
const micStateChip = document.getElementById("micStateChip");
const ttsToggleBtn = document.getElementById("ttsToggleBtn");
const ttsStopBtn = document.getElementById("ttsStopBtn");
const voiceParticleField = document.getElementById("voiceParticleField");
const commsLearning = document.getElementById("commsLearning");
const moodDisplay = document.getElementById("moodDisplay");
const moodEmoji = document.getElementById("moodEmoji");
const moodLabel = document.getElementById("moodLabel");
const moodMeta = document.getElementById("moodMeta");
const cameraToggleBtn = document.getElementById("cameraToggleBtn");
const moodSummaryBtn = document.getElementById("moodSummaryBtn");
const camStatusText = document.getElementById("camStatusText");
const detectedObjects = document.getElementById("detectedObjects");
const objectDetail = document.getElementById("objectDetail");
const visualTimeline = document.getElementById("visualTimeline");
const cameraFeed = document.getElementById("cameraFeed");

const wakePhrase = "hey jarvis";
let inputMode = "listen";
let recognition = null;
let ttsEnabled = true;
let preferredVoice = null;
let listenLoopEnabled = false;
let recognitionActive = false;
let recognitionRestartTimer = null;
const LISTEN_WATCHDOG_MS = 1200;
let recognitionRetryDelayMs = 260;
const RECOGNITION_RETRY_MAX_MS = 2200;
let micStream = null;
let lastMicErrorLogAt = 0;
let listenArmTimer = null;
const WEATHER_CACHE_KEY = "jarvis.weather.latest";
const WEATHER_RETRY_MS = 15000;
let weatherRetryTimer = null;
let lastWeatherLocationText = "Detecting location...";
let clientLocationHint = null;
// ...existing code...
let speakingActive = false;
let speechStartedAtMs = 0;
let voiceEnergy = 0.12;
let voiceOrbBlobs = [];
let micAudioContext = null;
let micAnalyser = null;
let micLevelBuffer = null;
let voiceOrbCanvas = null;
let voiceOrbCtx = null;
let voiceOrbDpr = 1;
let voiceFrameId = 0;

const bootTime = Date.now();
const learningUiState = {
  enabled: null,
  lastIntent: "idle",
  lastConfidence: null,
  profile: null,
  reasoningMode: "-",
};

const detectionState = {
  mood: "neutral",
  moodSymbol: "◉",
  moodConfidence: 0,
  brightness: 0,
  smoothedBrightness: null,
  stableMood: "neutral",
  detectedClasses: [],
  cocoModel: null,
  cocoReady: false,
  mobilenetModel: null,
  mobilenetReady: false,
  mobilenetLoading: false,
  captionModel: null,
  captionReady: false,
  captionLoading: false,
  faceApiReady: false,
  faceModelLoading: false,
  cameraActive: false,
  modelLoading: false,
  running: false,
  enabled: true,
  timerId: null,
  stream: null,
  peopleCount: 0,
  faceCount: 0,
  dominantExpression: "",
  sceneConfidence: 0,
  objectDetailText: "",
  captionText: "",
  frameDescription: "",
  lastVisionModel: "",
  lastVisionConfidence: 0,
  periodicVisionEnabled: true,
  moodHistory: [],
  visualEvents: [],
  lastAnnouncedMood: "",
  lastAnnouncedObjectsSig: "",
  lastAutoSummaryAt: 0,
  tinyCanvas: null,
  tinyCtx: null,
  lastDetectionTime: 0,
};
let lastUserInteractionAt = Date.now();

// Detection system functions
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`script load failed: ${src}`));
    document.head.appendChild(script);
  });
}

async function lazyLoadObjectModel() {
  if (detectionState.cocoReady || detectionState.modelLoading) {
    return;
  }
  detectionState.modelLoading = true;
  try {
    await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.0.0");
    await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3");
    if (window.cocoSsd) {
      detectionState.cocoModel = await window.cocoSsd.load({ base: "lite_mobilenet_v2" });
      detectionState.cocoReady = true;
    }
  } catch (_err) {
    detectionState.cocoReady = false;
  } finally {
    detectionState.modelLoading = false;
  }
}

async function lazyLoadFrameClassifier() {
  if (detectionState.mobilenetReady || detectionState.mobilenetLoading) {
    return;
  }
  detectionState.mobilenetLoading = true;
  try {
    await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.0.0");
    await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.1");
    if (window.mobilenet) {
      detectionState.mobilenetModel = await window.mobilenet.load({ version: 2, alpha: 1.0 });
      detectionState.mobilenetReady = true;
    }
  } catch (_err) {
    detectionState.mobilenetReady = false;
  } finally {
    detectionState.mobilenetLoading = false;
  }
}

async function lazyLoadCaptionModel() {
  if (detectionState.captionReady || detectionState.captionLoading) {
    return;
  }
  detectionState.captionLoading = true;
  try {
    if (!window.transformers && !window.pipeline) {
      await loadScript("https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js");
    }
    const pipelineFactory = window.transformers?.pipeline || window.pipeline;
    if (!pipelineFactory) {
      throw new Error("transformers pipeline unavailable");
    }
    detectionState.captionModel = await pipelineFactory("image-to-text", "Xenova/blip-image-captioning-base", {
      device: "wasm",
    });
    detectionState.captionReady = true;
  } catch (_err) {
    detectionState.captionReady = false;
  } finally {
    detectionState.captionLoading = false;
  }
}

async function lazyLoadFaceModels() {
  if (detectionState.faceApiReady || detectionState.faceModelLoading) {
    return;
  }
  detectionState.faceModelLoading = true;
  try {
    if (!window.faceapi) {
      await loadScript("https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js");
    }
    const modelBase = "https://justadudewhohacks.github.io/face-api.js/models";
    await window.faceapi.nets.tinyFaceDetector.loadFromUri(modelBase);
    await window.faceapi.nets.faceExpressionNet.loadFromUri(modelBase);
    detectionState.faceApiReady = true;
  } catch (_err) {
    detectionState.faceApiReady = false;
  } finally {
    detectionState.faceModelLoading = false;
  }
}

function renderVisualTimeline() {
  if (!visualTimeline) {
    return;
  }
  if (!detectionState.visualEvents.length) {
    visualTimeline.innerHTML = '<div class="event-row">No visual events yet</div>';
    return;
  }
  visualTimeline.innerHTML = detectionState.visualEvents
    .slice()
    .reverse()
    .map((evt) => `<div class="event-row">${evt}</div>`)
    .join("");
}

function recordVisualEvent(moodData, objects, peopleCount) {
  const stamp = new Date().toLocaleTimeString("en-US", { hour12: false, minute: "2-digit", second: "2-digit" });
  const objs = Array.isArray(objects) && objects.length ? objects.slice(0, 3).join(",") : "none";
  const row = `${stamp} | ${String(moodData.mood || "neutral").toUpperCase()} | people ${peopleCount} | ${objs}`;
  detectionState.visualEvents.push(row);
  if (detectionState.visualEvents.length > 10) {
    detectionState.visualEvents.shift();
  }
  renderVisualTimeline();
}

async function initCameraDetection() {
  if (!detectionState.enabled) {
    return;
  }
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.log("Camera not available");
      return;
    }
    
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
    if (cameraFeed) {
      cameraFeed.srcObject = stream;
      cameraFeed.play();
      detectionState.cameraActive = true;
      detectionState.stream = stream;
    }

    detectionState.tinyCanvas = document.createElement("canvas");
    detectionState.tinyCanvas.width = 48;
    detectionState.tinyCanvas.height = 36;
    detectionState.tinyCtx = detectionState.tinyCanvas.getContext("2d", { willReadFrequently: true });

    // Defer heavy model loading until UI is stable.
    window.setTimeout(() => {
      lazyLoadObjectModel();
      lazyLoadFaceModels();
      lazyLoadFrameClassifier();
      lazyLoadCaptionModel();
    }, 2200);

    if (detectionState.timerId) {
      clearInterval(detectionState.timerId);
    }
    detectionState.timerId = setInterval(runDetection, 20000);
    window.setTimeout(runDetection, 1200);
    updateCameraStatusUi();

    // --- New: On first camera ready, capture and send frame to backend vision model ---
    // Wait for video to be ready
    function onCameraReadyOnce() {
      if (cameraFeed && cameraFeed.readyState === cameraFeed.HAVE_ENOUGH_DATA) {
        // Capture frame and send to backend
        const imageDataUrl = captureCurrentFrameDataUrl();
        if (imageDataUrl) {
          requestBackendVisionDetail(imageDataUrl, "Describe the scene", "scene").then((result) => {
            if (result && result.ok && result.description) {
              detectionState.objectDetailText = result.description;
              detectionState.frameDescription = result.description;
              if (objectDetail) objectDetail.textContent = result.description;
            } else if (objectDetail) {
              objectDetail.textContent = result && result.error ? result.error : "Vision model did not return a description.";
            }
          });
        }
        cameraFeed.removeEventListener("playing", onCameraReadyOnce);
      }
    }
    cameraFeed.addEventListener("playing", onCameraReadyOnce);
    // If already ready, trigger immediately
    if (cameraFeed && cameraFeed.readyState === cameraFeed.HAVE_ENOUGH_DATA) {
      onCameraReadyOnce();
    }
  } catch (err) {
    console.log("Camera permission denied or unavailable:", err.message);
    updateMoodDisplay({
      mood: "camera blocked",
      symbol: "◍",
      confidence: 0,
      brightness: 0,
      mode: "permission",
    });
    if (camStatusText) {
      camStatusText.textContent = "Camera permission denied";
    }
  }
}

function updateCameraStatusUi() {
  if (cameraToggleBtn) {
    const on = detectionState.enabled && detectionState.cameraActive;
    cameraToggleBtn.textContent = on ? "CAMERA: ON" : "CAMERA: OFF";
    cameraToggleBtn.classList.toggle("off", !on);
  }
  if (camStatusText) {
    if (!detectionState.enabled) {
      camStatusText.textContent = "Camera paused by user";
    } else if (detectionState.cameraActive) {
      const faceState = detectionState.faceApiReady ? "face ready" : (detectionState.faceModelLoading ? "face loading" : "face pending");
      const captionState = detectionState.captionReady ? "caption ready" : (detectionState.captionLoading ? "caption loading" : "caption pending");
      const modelState = detectionState.lastVisionModel
        ? `vision ${detectionState.lastVisionModel}${detectionState.lastVisionConfidence ? ` (${Math.round(detectionState.lastVisionConfidence * 100)}%)` : ""}`
        : "vision endpoint pending";
      const periodicState = detectionState.periodicVisionEnabled ? "hand/object scan 20s" : "manual object scan";
      camStatusText.textContent = `Live scan every 20s | ${periodicState} | ${faceState} | ${captionState} | ${modelState}`;
    } else {
      camStatusText.textContent = "Camera initializing...";
    }
  }
}

function stopCameraDetection() {
  detectionState.enabled = false;
  if (detectionState.timerId) {
    clearInterval(detectionState.timerId);
    detectionState.timerId = null;
  }
  if (detectionState.stream) {
    detectionState.stream.getTracks().forEach((track) => track.stop());
    detectionState.stream = null;
  }
  if (cameraFeed) {
    cameraFeed.srcObject = null;
  }
  detectionState.cameraActive = false;
  updateDetectedObjectsDisplay([], false);
  updateMoodDisplay({
    mood: "camera off",
    symbol: "◍",
    confidence: 0,
    brightness: detectionState.brightness || 0,
    mode: "paused",
  });
  updateCameraStatusUi();
}

async function toggleCameraDetection() {
  if (detectionState.enabled && detectionState.cameraActive) {
    stopCameraDetection();
    return;
  }
  detectionState.enabled = true;
  updateCameraStatusUi();
  await initCameraDetection();
}

function summarizeMoodHistory() {
  const items = detectionState.moodHistory;
  if (!items.length) {
    const msg = "Mood summary: no data yet.";
    addLogRow("ai", msg, "camera detection");
    return msg;
  }
  const counts = { focused: 0, neutral: 0, alert: 0, other: 0 };
  let confSum = 0;
  items.forEach((it) => {
    const key = Object.prototype.hasOwnProperty.call(counts, it.mood) ? it.mood : "other";
    counts[key] += 1;
    confSum += Number(it.confidence || 0);
  });
  const total = items.length;
  const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0].toUpperCase();
  const avgConf = Math.round((confSum / total) * 100);
  const latestObjects = detectionState.detectedClasses.length ? detectionState.detectedClasses.join(", ") : "none";
  const people = Number(detectionState.peopleCount || 0);
  const summary = `Mood summary (${total} scans): Dominant ${dominant}, avg signal ${avgConf}%, people ${people}, latest objects ${latestObjects}.`;
  addLogRow("ai", summary, "mood summary");
  return summary;
}

function summarizeVisualScene() {
  const mood = String(detectionState.mood || "neutral");
  const signal = Math.round(Number(detectionState.moodConfidence || 0) * 100);
  const objects = detectionState.detectedClasses.length ? detectionState.detectedClasses.join(", ") : "none";
  const people = Number(detectionState.peopleCount || 0);
  const scanAgeSec = detectionState.lastDetectionTime ? Math.max(0, Math.round((Date.now() - detectionState.lastDetectionTime) / 1000)) : null;
  const scanText = scanAgeSec === null ? "no scan yet" : `${scanAgeSec}s ago`;
  const cameraText = detectionState.enabled && detectionState.cameraActive ? "camera on" : "camera off";
  return `Scene summary: ${cameraText}, mood ${mood} (${signal}% signal), people ${people}, detected objects ${objects}, last scan ${scanText}.`;
}

function sleepMs(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function captureCurrentFrameDataUrl() {
  const frameCanvas = captureFrameCanvas();
  if (!frameCanvas) {
    return "";
  }
  try {
    // Resize to max 512px wide before encoding — reduces payload by ~70%
    const MAX_W = 512;
    const srcW = frameCanvas.width || 1;
    const srcH = frameCanvas.height || 1;
    const scale = srcW > MAX_W ? MAX_W / srcW : 1;
    const outW = Math.round(srcW * scale);
    const outH = Math.round(srcH * scale);
    if (scale === 1) {
      return frameCanvas.toDataURL("image/jpeg", 0.75);
    }
    const small = document.createElement("canvas");
    small.width = outW;
    small.height = outH;
    const ctx = small.getContext("2d");
    ctx.drawImage(frameCanvas, 0, 0, outW, outH);
    return small.toDataURL("image/jpeg", 0.75);
  } catch (_err) {
    return "";
  }
}

async function requestBackendVisionDetail(imageDataUrl, queryText, mode = "hand") {
  if (!imageDataUrl) {
    return { ok: false, error: "frame unavailable" };
  }

  const prompt = mode === "hand"
    ? `User asked: ${queryText || "what is in my hand"}. From this full frame, identify the object in hand and then describe all other visible objects, including small background items. Give a clear multi-line description with object names, rough position (left/center/right), color/shape cues, and confidence words like high/medium/low where useful.`
    : `User asked: ${queryText || "describe current frame"}. Describe all visible objects in this full frame, both big and small. Provide a rich multi-line scene description with object names, relative position, and visual cues.`;

  try {
    const resp = await fetch("/api/vision-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_data_url: imageDataUrl, prompt, mode }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      return { ok: false, error: data.error || "vision endpoint unavailable" };
    }
    return {
      ok: true,
      description: String(data.description || "").trim(),
      model: String(data.model || ""),
      confidence: Number(data.confidence || 0),
    };
  } catch (err) {
    return { ok: false, error: String(err?.message || err || "vision request failed") };
  }
}

function hasReliableObjectDetail() {
  const detail = String(detectionState.objectDetailText || "").toLowerCase();
  if (!detail) {
    return false;
  }
  return !(
    detail.includes("no frame description yet") ||
    detail.includes("waiting for frame analysis")
  );
}

async function performActiveVisualCheck(queryText) {
  const query = String(queryText || "").toLowerCase();
  const asksHand = /(hand|palm|holding|in my hand|on my hand|in my palm|what am i showing|what did i show|what i am holding)/i.test(query);

  if (!(detectionState.enabled && detectionState.cameraActive)) {
    detectionState.enabled = true;
    await initCameraDetection();
    await sleepMs(700);
  }

  await Promise.allSettled([
    lazyLoadObjectModel(),
    lazyLoadFrameClassifier(),
    lazyLoadCaptionModel(),
  ]);

  addLogRow("ai", "Understood. Hold the object steady, capturing frame in 2 seconds...", "active visual check");
  await sleepMs(2000);
  const frameDataUrl = captureCurrentFrameDataUrl();
  const backendVision = await requestBackendVisionDetail(frameDataUrl, queryText, asksHand ? "hand" : "scene");

  if (backendVision.ok && backendVision.description) {
    detectionState.lastVisionModel = backendVision.model;
    detectionState.lastVisionConfidence = backendVision.confidence;
    detectionState.objectDetailText = backendVision.description;
    detectionState.captionText = backendVision.description;
    detectionState.frameDescription = backendVision.description;
    if (objectDetail) {
      objectDetail.textContent = backendVision.description;
    }
  } else {
    await updateFrameDescriptionFromSnapshot(asksHand ? "hand" : "scene");
  }
  await runDetection();

  let response = "I checked the frame but could not get a reliable object yet.";
  if (backendVision.ok && backendVision.description) {
    response = asksHand
      ? `I checked your hand request. ${backendVision.description}`
      : `I checked the frame. ${backendVision.description}`;
  } else if (hasReliableObjectDetail()) {
    const firstLine = String(detectionState.objectDetailText || "").split("\n")[0];
    response = asksHand
      ? `I checked your hand request. ${firstLine}`
      : `I checked the frame. ${firstLine}`;
  } else if (detectionState.detectedClasses.length) {
    response = `I checked the frame. I can detect ${detectionState.detectedClasses.slice(0, 5).join(", ")}.`;
  } else if (asksHand) {
    response = "I checked after 2 seconds but still cannot read the object clearly. Hold it closer to camera and keep it steady.";
  }

  if (!backendVision.ok && backendVision.error && /api key|unavailable|failed|not configured/i.test(String(backendVision.error))) {
    response += " Strong vision model is unavailable right now, so this answer used local fallback analysis.";
  }

  addLogRow("ai", response, "active visual check");
  speakText(response);
}

async function handleVisualQuickCommand(rawText) {
  const text = String(rawText || "").trim().toLowerCase();
  const normalized = text
    .replace(/[^a-z0-9\s']/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const plain = normalized.replace(/'/g, "");
  if (!text) {
    return false;
  }

  const forceHandObjectCheck = /\b(whats|what\s+is|what\s+s|what)\s+(?:in|on)\s+my\s+(?:hand|palm)\b/.test(plain)
    || /\bwhat\s+am\s+i\s+holding\b/.test(plain);

  const isMoodSummary = /(summarize mood|mood summary|summary of mood|how is my mood|my mood summary)/i.test(text);
  const isSceneSummary = /(what(?:'s|s| is)?\s+go+i+n+g\s+on|what\s+going\s+on|what\s+do\s+you\s+see|scene\s+summary|summari[sz]e\s+scene|visual\s+summary|what\s+is\s+in\s+frame)/i.test(text);
  const directObjectCheck = /(what(?:['’]?s| is)?\s+(?:in|on)\s+my\s+(?:hand|palm)|what\s+am\s+i\s+holding|what\s+do\s+i\s+have\s+in\s+my\s+hand|what\s+is\s+in\s+my\s+palm|what\s+am\s+i\s+showing|what\s+did\s+i\s+show\s+you|can\s+you\s+identify\s+this|identify\s+(?:this\s+)?(?:item|object)|check\s+my\s+(?:hand|palm)|detect\s+(?:the\s+)?(?:item|object)|check\s+(?:the\s+)?(?:item|object)|look\s+at\s+my\s+(?:hand|palm)|see\s+my\s+(?:hand|palm)|tell\s+me\s+what\s+this\s+is|what\s+is\s+this\s+in\s+my\s+hand)/i.test(normalized);
  const hasObjectIntentVerb = /\b(what|whats|which|identify|check|detect|tell|name|guess|recognize|recognise|find)\b/.test(normalized);
  const hasHandContext = /\b(hand|palm|holding|hold|showing|show|item|object|thing|this|it)\b/.test(normalized);
  const fuzzyObjectCheck = hasObjectIntentVerb && hasHandContext;
  const isActiveObjectCheck = directObjectCheck || fuzzyObjectCheck;
  const isCameraOn = /(camera on|turn camera on|start camera|enable camera)/i.test(text);
  const isCameraOff = /(camera off|turn camera off|stop camera|disable camera)/i.test(text);

  if (forceHandObjectCheck || isActiveObjectCheck) {
    await performActiveVisualCheck(text);
    return true;
  }

  if (isMoodSummary) {
    const summary = summarizeMoodHistory();
    speakText(summary);
    return true;
  }

  if (isSceneSummary) {
    const summary = summarizeVisualScene();
    addLogRow("ai", summary, "scene summary");
    speakText(summary);
    return true;
  }

  if (isCameraOn) {
    if (!(detectionState.enabled && detectionState.cameraActive)) {
      await toggleCameraDetection();
    }
    const msg = "Camera is on. Visual detection is active.";
    addLogRow("ai", msg, "camera control");
    speakText(msg);
    return true;
  }

  if (isCameraOff) {
    if (detectionState.enabled && detectionState.cameraActive) {
      stopCameraDetection();
    }
    const msg = "Camera is off. Visual detection is paused.";
    addLogRow("ai", msg, "camera control");
    speakText(msg);
    return true;
  }

  return false;
}

function maybeAutoSpeakDetection(moodData, detectedClasses) {
  const now = Date.now();
  const isIdle = now - lastUserInteractionAt > 12000;
  const cooldownOk = now - detectionState.lastAutoSummaryAt > 50000;
  const objects = Array.isArray(detectedClasses) ? detectedClasses : [];
  const objectsSig = objects.slice().sort().join("|");
  const mood = String(moodData?.mood || "neutral");
  const moodChanged = mood !== detectionState.lastAnnouncedMood;
  const objectsChanged = objectsSig !== detectionState.lastAnnouncedObjectsSig;
  const moodConfidence = Number(moodData?.confidence || 0);
  const hasSignal = objects.length > 0 || moodConfidence >= 0.38;
  const significantObjectChange = Math.abs(objects.length - (detectionState.lastAnnouncedObjectsSig ? detectionState.lastAnnouncedObjectsSig.split("|").filter(Boolean).length : 0)) >= 1;
  const significantChange = moodChanged || (objectsChanged && significantObjectChange);

  if (!isIdle || !cooldownOk || !hasSignal) {
    return;
  }
  if (!significantChange) {
    return;
  }
  if (recognitionActive || speakingActive) {
    return;
  }

  const people = Number(detectionState.peopleCount || 0);
  const moodText = `User mood looks ${mood}`;
  const peopleText = people > 1 ? `${people} people are in frame` : (people === 1 ? "one person is in frame" : "no person detected");
  const objectText = detectionState.objectDetailText
    ? detectionState.objectDetailText.split("\n")[0]
    : (objects.length ? `I can see ${objects.slice(0, 4).join(", ")}` : "No major object change in frame");
  const summary = `Visual update: ${moodText}. ${objectText}.`;
  const fullSummary = `${summary} Also, ${peopleText}.`;
  addLogRow("ai", fullSummary, "auto visual summary");
  speakText(fullSummary);

  detectionState.lastAnnouncedMood = mood;
  detectionState.lastAnnouncedObjectsSig = objectsSig;
  detectionState.lastAutoSummaryAt = now;
}

async function runDetection() {
  if (!detectionState.enabled || !detectionState.cameraActive || !cameraFeed || detectionState.running || document.visibilityState !== "visible") return;
  detectionState.running = true;

  try {
    const faceMoodData = await analyzeMoodFromFace();
    const moodData = faceMoodData || analyzeMoodFromFrame();
    updateMoodDisplay(moodData);
    detectionState.lastDetectionTime = Date.now();
    detectionState.moodHistory.push({
      mood: moodData.mood,
      confidence: Number(moodData.confidence || 0),
      t: detectionState.lastDetectionTime,
    });
    if (detectionState.moodHistory.length > 12) {
      detectionState.moodHistory.shift();
    }

    if (detectionState.cocoReady && detectionState.cocoModel && cameraFeed.readyState === cameraFeed.HAVE_ENOUGH_DATA) {
      const predictions = await detectionState.cocoModel.detect(cameraFeed);
      const sorted = predictions
        .filter((p) => Number(p.score || 0) >= 0.35)
        .sort((a, b) => Number(b.score || 0) - Number(a.score || 0));
      const topClasses = [...new Set(sorted.map((p) => p.class))].slice(0, 8);
      const personDetections = sorted.filter((p) => String(p.class || "").toLowerCase() === "person");
      detectionState.peopleCount = Math.max(detectionState.faceCount, personDetections.length);
      detectionState.detectedClasses = topClasses;
      updateDetectedObjectsDisplay(topClasses, false);
      await updateFrameDescriptionFromSnapshot("scene");
    } else {
      updateDetectedObjectsDisplay(detectionState.detectedClasses, true);
      if (detectionState.captionReady) {
        await updateFrameDescriptionFromSnapshot("scene");
      }
    }

    // Default periodic hand/object model check on every 20s scan cycle.
    if (detectionState.periodicVisionEnabled) {
      const periodicFrame = captureCurrentFrameDataUrl();
      if (periodicFrame) {
        const periodicVision = await requestBackendVisionDetail(
          periodicFrame,
          "Periodic scan: describe all visible objects in frame including small items and background details.",
          "scene"
        );
        if (periodicVision.ok && periodicVision.description) {
          detectionState.lastVisionModel = periodicVision.model;
          detectionState.lastVisionConfidence = Number(periodicVision.confidence || 0);
          detectionState.objectDetailText = periodicVision.description;
          detectionState.captionText = periodicVision.description;
          detectionState.frameDescription = periodicVision.description;
          if (objectDetail) {
            objectDetail.textContent = periodicVision.description;
          }
        }
      }
    }

    detectionState.sceneConfidence = Math.max(Number(moodData.confidence || 0), detectionState.detectedClasses.length ? 0.55 : 0.2);
    recordVisualEvent(moodData, detectionState.detectedClasses, detectionState.peopleCount);

    maybeAutoSpeakDetection(moodData, detectionState.detectedClasses);

    // Send to backend for conversation integration
    await sendDetectionToBackend({
      mood: detectionState.mood,
      mood_confidence: detectionState.moodConfidence,
      scene_confidence: detectionState.sceneConfidence,
      people_count: detectionState.peopleCount,
      face_count: detectionState.faceCount,
      dominant_expression: detectionState.dominantExpression,
      object_detail: detectionState.objectDetailText,
      caption_text: detectionState.captionText,
      frame_description: detectionState.frameDescription,
      objects: detectionState.detectedClasses,
      timeline: detectionState.visualEvents.slice(-6),
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    console.log("Detection error:", err.message);
  } finally {
    detectionState.running = false;
  }
}

function captureFrameCanvas() {
  if (!cameraFeed || cameraFeed.readyState !== cameraFeed.HAVE_ENOUGH_DATA) {
    return null;
  }
  const width = cameraFeed.videoWidth || 0;
  const height = cameraFeed.videoHeight || 0;
  if (!width || !height) {
    return null;
  }
  const frameCanvas = document.createElement("canvas");
  frameCanvas.width = width;
  frameCanvas.height = height;
  const frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });
  frameCtx.drawImage(cameraFeed, 0, 0, width, height);
  return frameCanvas;
}

function getDominantColorName(canvas) {
  if (!canvas) {
    return "unknown";
  }
  try {
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let r = 0;
    let g = 0;
    let b = 0;
    let count = 0;
    for (let i = 0; i < data.length; i += 16) {
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      count += 1;
    }
    if (!count) {
      return "unknown";
    }
    r /= count;
    g /= count;
    b /= count;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const chroma = max - min;
    const brightness = (r + g + b) / 3;

    if (brightness > 220 && chroma < 20) return "white";
    if (brightness > 160 && chroma < 22) return "silver";
    if (brightness < 50 && chroma < 20) return "black";
    if (r > g * 1.25 && r > b * 1.25) return "red";
    if (g > r * 1.15 && g > b * 1.15) return "green";
    if (b > r * 1.15 && b > g * 1.15) return "blue";
    if (r > 120 && g > 90 && b < 90) return "yellow";
    if (r > 100 && g > 60 && b < 60) return "orange";
    if (r > 70 && g > 40 && b < 35) return "brown";
    if (chroma < 18) return brightness > 140 ? "silver" : "gray";
    return "multicolor";
  } catch (_err) {
    return "unknown";
  }
}

function makeArticle(word) {
  const first = String(word || "").trim().charAt(0).toLowerCase();
  return /[aeiou]/.test(first) ? "an" : "a";
}

function isGenericCaption(captionText) {
  const t = String(captionText || "").trim().toLowerCase();
  if (!t) {
    return true;
  }
  return (
    t === "object" ||
    t === "a object" ||
    t === "an object" ||
    /\b(black|white|gray|silver)\s+object\b/.test(t) ||
    /\b(an?|the)\s+(thing|item|object)\b/.test(t)
  );
}

function cleanCaption(captionText) {
  return String(captionText || "")
    .replace(/^\s*(this is|there is|it is)\s+/i, "")
    .replace(/\s+/g, " ")
    .trim();
}

function inferNounFromCaption(captionText) {
  const raw = String(captionText || "").toLowerCase();
  if (!raw) {
    return "";
  }
  const aliasMap = {
    smartphone: "phone",
    mobile: "phone",
    cellphone: "phone",
    "cell phone": "phone",
    telephone: "phone",
    lighter: "lighter",
    igniter: "lighter",
    mug: "mug",
    cup: "mug",
    bottle: "bottle",
    remote: "remote",
    wallet: "wallet",
    keychain: "keys",
    key: "keys",
    knife: "knife",
    spoon: "spoon",
  };

  const candidates = raw
    .replace(/[.;:!?]/g, ",")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);

  const joined = ` ${raw} `;
  for (const [k, v] of Object.entries(aliasMap)) {
    if (joined.includes(` ${k} `)) {
      return v;
    }
  }

  for (const token of candidates) {
    const t = token.replace(/\b(a|an|the)\b/g, "").trim();
    if (!t || /(object|thing|item)/.test(t)) {
      continue;
    }
    const words = t.split(/\s+/);
    const last = words[words.length - 1];
    if (last && !/(black|white|gray|silver|light|dark)/.test(last)) {
      return aliasMap[last] || last;
    }
  }

  return "";
}

function buildRichObjectDescription(className, captionText, colorName, objectHints = []) {
  const normalizedClass = String(className || "object").toLowerCase();
  const caption = cleanCaption(String(captionText || "").replace(/[.]+$/, ""));
  const color = String(colorName || "").trim().toLowerCase();
  const classAliasMap = {
    cup: "mug",
    keyboard: "keyboard",
    cell_phone: "phone",
    tv: "screen",
    pottedplant: "plant",
  };
  const hinted = Array.isArray(objectHints) && objectHints.length ? String(objectHints[0] || "").toLowerCase() : "";
  const hintedNoun = hinted ? (classAliasMap[hinted] || hinted) : "";
  const captionNoun = inferNounFromCaption(caption);
  const noun = normalizedClass === "object"
    ? (hintedNoun || captionNoun || "item")
    : (classAliasMap[normalizedClass] || normalizedClass);
  const colorPrefix = color && color !== "unknown" && color !== "multicolor" ? `${color} ` : "";
  const baseSentence = `This looks like ${makeArticle(`${colorPrefix}${noun}`)} ${colorPrefix}${noun}`.replace(/\s+/g, " ");
  const handleHint = /\b(handle|cup|mug)\b/i.test(`${caption} ${normalizedClass}`) && /\b(mug|cup)\b/i.test(noun);
  const finalSentence = `${baseSentence}${handleHint ? " with a handle" : ""}.`;
  const inferred = inferNounFromCaption(caption);
  const captionLine = caption && !isGenericCaption(caption)
    ? `Caption: ${caption}.`
    : (hintedNoun ? `Model hint: likely ${hintedNoun}.` : (inferred ? `Model hint: likely ${inferred}.` : "Caption: not available yet."));
  return `${finalSentence}\n${captionLine}`;
}

async function updateFrameDescriptionFromSnapshot(mode = "scene") {
  if (!objectDetail || !cameraFeed || cameraFeed.readyState !== cameraFeed.HAVE_ENOUGH_DATA) {
    return;
  }

  const frameCanvas = captureFrameCanvas();
  if (!frameCanvas) {
    return;
  }

  const dominantColor = getDominantColorName(frameCanvas);
  const hints = (detectionState.detectedClasses || []).slice(0, 3);
  let captionText = "";
  if (detectionState.captionReady && detectionState.captionModel) {
    try {
      const captionResult = await detectionState.captionModel(frameCanvas, {
        max_new_tokens: 28,
        min_length: 8,
      });
      const rawCaption = Array.isArray(captionResult) ? captionResult[0]?.generated_text : captionResult?.generated_text;
      if (rawCaption) {
        captionText = String(rawCaption).trim();
      }
    } catch (_err) {
      // Ignore and fall back.
    }
  }

  if (!captionText && detectionState.mobilenetReady && detectionState.mobilenetModel) {
    try {
      const classifications = await detectionState.mobilenetModel.classify(frameCanvas, 2);
      if (Array.isArray(classifications) && classifications.length) {
        captionText = String(classifications[0].className || "").trim();
      }
    } catch (_err) {
      // Ignore and leave blank.
    }
  }

  const prefix = mode === "hand" ? "Hand check" : "Frame check";
  if (captionText) {
    const desc = buildRichObjectDescription("object", captionText, dominantColor, hints);
    detectionState.objectDetailText = `${prefix}: ${desc}\nColor hint: ${dominantColor}`;
  } else {
    const fallbackHint = hints.length ? `likely ${hints[0]}` : "no frame description yet";
    detectionState.objectDetailText = `${prefix}: ${fallbackHint}.`;
  }
  detectionState.captionText = captionText;
  detectionState.frameDescription = detectionState.objectDetailText;
  objectDetail.textContent = detectionState.objectDetailText;
}

async function analyzeMoodFromFace() {
  if (!detectionState.faceApiReady || !window.faceapi || !cameraFeed || cameraFeed.readyState !== cameraFeed.HAVE_ENOUGH_DATA) {
    return null;
  }
  try {
    const detections = await window.faceapi
      .detectAllFaces(cameraFeed, new window.faceapi.TinyFaceDetectorOptions({ inputSize: 160, scoreThreshold: 0.4 }))
      .withFaceExpressions();
    detectionState.faceCount = Array.isArray(detections) ? detections.length : 0;
    if (!detections || !detections.length) {
      detectionState.dominantExpression = "";
      return null;
    }

    const expressionTotals = { neutral: 0, happy: 0, sad: 0, angry: 0, fearful: 0, disgusted: 0, surprised: 0 };
    detections.forEach((d) => {
      const expr = d.expressions || {};
      Object.keys(expressionTotals).forEach((k) => {
        expressionTotals[k] += Number(expr[k] || 0);
      });
    });
    const dominant = Object.entries(expressionTotals).sort((a, b) => b[1] - a[1])[0];
    const dominantKey = String(dominant[0] || "neutral");
    const dominantValue = Number(dominant[1] || 0) / detections.length;
    detectionState.dominantExpression = dominantKey;

    let mood = "neutral";
    let symbol = "◉";
    if (dominantKey === "happy") {
      mood = "focused";
      symbol = "◈";
    } else if (["angry", "fearful", "surprised"].includes(dominantKey)) {
      mood = "alert";
      symbol = "◎";
    } else if (["sad", "disgusted"].includes(dominantKey)) {
      mood = "stressed";
      symbol = "◌";
    }

    detectionState.mood = mood;
    detectionState.moodSymbol = symbol;
    detectionState.moodConfidence = Math.min(1, Math.max(0, dominantValue));
    return {
      mood,
      symbol,
      confidence: detectionState.moodConfidence,
      brightness: Number(detectionState.brightness || 0),
      mode: "face",
      people: detections.length,
      expression: dominantKey,
    };
  } catch (_err) {
    return null;
  }
}

function analyzeMoodFromFrame() {
  if (!cameraFeed || cameraFeed.readyState !== cameraFeed.HAVE_ENOUGH_DATA || !detectionState.tinyCtx || !detectionState.tinyCanvas) {
    return { mood: "neutral", symbol: "◉", confidence: 0, brightness: 0, mode: "standby" };
  }

  try {
    const canvas = detectionState.tinyCanvas;
    const ctx = detectionState.tinyCtx;
    ctx.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);

    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    let brightness = 0;

    for (let i = 0; i < data.length; i += 4) {
      brightness += (data[i] + data[i + 1] + data[i + 2]) / 3;
    }
    brightness /= (canvas.width * canvas.height);

    if (detectionState.smoothedBrightness === null) {
      detectionState.smoothedBrightness = brightness;
    } else {
      detectionState.smoothedBrightness = detectionState.smoothedBrightness * 0.72 + brightness * 0.28;
    }

    const b = detectionState.smoothedBrightness;
    let mood = detectionState.stableMood;
    // Hysteresis bands prevent rapid mood toggling around thresholds.
    if (detectionState.stableMood === "focused") {
      if (b > 92) {
        mood = b > 188 ? "alert" : "neutral";
      }
    } else if (detectionState.stableMood === "alert") {
      if (b < 168) {
        mood = b < 72 ? "focused" : "neutral";
      }
    } else {
      if (b < 72) {
        mood = "focused";
      } else if (b > 188) {
        mood = "alert";
      } else {
        mood = "neutral";
      }
    }
    detectionState.stableMood = mood;

    let symbol;
    let confidence;
    if (mood === "focused") {
      symbol = "◈";
      confidence = Math.min(1, Math.max(0, (100 - b) / 55));
    } else if (mood === "alert") {
      symbol = "◎";
      confidence = Math.min(1, Math.max(0, (b - 160) / 65));
    } else {
      symbol = "◉";
      confidence = 1 - Math.min(1, Math.abs(b - 130) / 70);
    }

    detectionState.mood = mood;
    detectionState.moodSymbol = symbol;
    detectionState.moodConfidence = confidence;
    detectionState.brightness = b;
    return { mood, symbol, confidence, brightness: b, mode: detectionState.cocoReady ? "full" : "light" };
  } catch (err) {
    return { mood: "neutral", symbol: "◉", confidence: 0, brightness: 0, mode: "error" };
  }
}

function updateMoodDisplay(moodData) {
  if (moodEmoji) {
    moodEmoji.textContent = moodData.symbol;
    moodEmoji.classList.remove("mood-focused", "mood-alert", "mood-neutral", "mood-camera-blocked");
    const cls = String(moodData.mood || "neutral").replace(/\s+/g, "-").toLowerCase();
    moodEmoji.classList.add(`mood-${cls}`);
  }
  if (moodLabel) {
    moodLabel.textContent = `USER MOOD: ${String(moodData.mood || "neutral").toUpperCase()}`;
  }
  if (moodMeta) {
    const confPct = Math.round(Number(moodData.confidence || 0) * 100);
    const bright = Math.round(Number(moodData.brightness || 0));
    const mode = String(moodData.mode || "light").toUpperCase();
    const people = Math.max(Number(detectionState.peopleCount || 0), Number(detectionState.faceCount || 0));
    const expr = detectionState.dominantExpression ? ` | Expr ${String(detectionState.dominantExpression).toUpperCase()}` : "";
    moodMeta.textContent = `Signal ${confPct}% | Lux ${bright} | ${mode} | People ${people}${expr}`;
  }
}

function updateDetectedObjectsDisplay(classes, modelPending = false) {
  if (!detectedObjects) return;
  if (!detectionState.enabled) {
    detectedObjects.textContent = "Detection paused";
    return;
  }
  if (!Array.isArray(classes) || classes.length === 0) {
    detectedObjects.textContent = modelPending ? "Scanning frame..." : "No objects detected in frame";
  } else {
    const people = Number(detectionState.peopleCount || 0);
    const peopleText = people > 1 ? `People in frame: ${people} | ` : (people === 1 ? "Person in frame | " : "");
    detectedObjects.textContent = `${peopleText}Detected: ${classes.join(", ")}`;
  }
}

async function sendDetectionToBackend(detectionData) {
  try {
    await fetch("/api/detection", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(detectionData),
    });
  } catch (err) {
    console.log("Detection backend sync failed:", err.message);
  }
}

function renderCommsLearning() {
  if (!commsLearning) {
    return;
  }
  const state = learningUiState.enabled === null ? "UNKNOWN" : (learningUiState.enabled ? "ON" : "OFF");
  const conf = learningUiState.lastConfidence === null ? "-" : Number(learningUiState.lastConfidence).toFixed(2);
  const profile = learningUiState.profile || {};
  const depth = Number(profile.technical_depth ?? 0.5).toFixed(2);
  const stress = Number(profile.stress_sensitivity ?? 0.5).toFixed(2);
  const formality = Number(profile.formality ?? 0.5).toFixed(2);
  commsLearning.textContent = [
    `Learning: ${state}`,
    `Last intent: ${learningUiState.lastIntent} (conf ${conf})`,
    `Depth ${depth} | StressSense ${stress}`,
    `Formality ${formality} | Mode ${learningUiState.reasoningMode || "-"}`,
  ].join("\n");
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function setupMicAnalyser(stream) {
  if (!stream || micAnalyser || !(window.AudioContext || window.webkitAudioContext)) {
    return;
  }
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    micAudioContext = new Ctx();
    const source = micAudioContext.createMediaStreamSource(stream);
    micAnalyser = micAudioContext.createAnalyser();
    micAnalyser.fftSize = 256;
    micLevelBuffer = new Uint8Array(micAnalyser.fftSize);
    source.connect(micAnalyser);
  } catch (_err) {
    micAudioContext = null;
    micAnalyser = null;
    micLevelBuffer = null;
  }
}

function sampleMicLevel() {
  if (!micAnalyser || !micLevelBuffer) {
    return 0;
  }
  micAnalyser.getByteTimeDomainData(micLevelBuffer);
  let sumSquares = 0;
  for (let i = 0; i < micLevelBuffer.length; i += 1) {
    const normalized = (micLevelBuffer[i] - 128) / 128;
    sumSquares += normalized * normalized;
  }
  const rms = Math.sqrt(sumSquares / micLevelBuffer.length);
  return clamp01(rms * 4.1);
}

function resizeVoiceOrbCanvas() {
  if (!voiceParticleField || !voiceOrbCanvas) {
    return;
  }
  const rect = voiceParticleField.getBoundingClientRect();
  const size = Math.max(24, Math.floor(Math.min(rect.width, rect.height)));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  voiceOrbDpr = dpr;
  voiceOrbCanvas.width = Math.floor(size * dpr);
  voiceOrbCanvas.height = Math.floor(size * dpr);
  voiceOrbCanvas.style.width = `${size}px`;
  voiceOrbCanvas.style.height = `${size}px`;
}

function initVoiceParticles() {
  if (!voiceParticleField) {
    return;
  }
  voiceParticleField.innerHTML = "";
  voiceOrbCanvas = document.createElement("canvas");
  voiceOrbCanvas.className = "voice-orb-canvas";
  voiceParticleField.appendChild(voiceOrbCanvas);
  voiceOrbCtx = voiceOrbCanvas.getContext("2d", { alpha: true });
  resizeVoiceOrbCanvas();

  voiceOrbBlobs = [];
  const count = 18;
  for (let i = 0; i < count; i += 1) {
    voiceOrbBlobs.push({
      angle: Math.random() * Math.PI * 2,
      orbit: 0.12 + Math.random() * 0.56,
      speed: 0.18 + Math.random() * 0.82,
      radius: 0.06 + Math.random() * 0.13,
      hueShift: (i * 31) % 360,
      drift: Math.random() * Math.PI * 2,
    });
  }

  window.addEventListener("resize", resizeVoiceOrbCanvas);
}

function animateVoiceParticles(nowMs = performance.now()) {
  if (!voiceParticleField || !voiceOrbCtx || !voiceOrbCanvas || !voiceOrbBlobs.length) {
    voiceFrameId = requestAnimationFrame(animateVoiceParticles);
    return;
  }

  let mode = "idle";
  let targetEnergy = 0.12;
  let rotationSpeed = 0.25;
  let baseHue = 195;
  let glowIntensity = 0.3;

  if (speakingActive) {
    mode = "speaking";
    const elapsedSec = Math.max(0, (nowMs - speechStartedAtMs) / 1000);
    const cadenceWave = 0.5 + 0.5 * Math.sin(elapsedSec * 8.2);
    targetEnergy = 0.7 + cadenceWave * 0.3;
    rotationSpeed = 1.5;
    baseHue = 310;
    glowIntensity = 0.5;
  } else if (recognitionActive) {
    mode = "listening";
    const micLevel = sampleMicLevel();
    targetEnergy = 0.35 + micLevel * 0.65;
    rotationSpeed = 0.8 + micLevel * 0.8;
    baseHue = 200 + micLevel * 30;
    glowIntensity = 0.35 + micLevel * 0.25;
  }

  voiceEnergy += (targetEnergy - voiceEnergy) * 0.09;
  const energy = clamp01(voiceEnergy);
  voiceParticleField.dataset.mode = mode;

  const ctx = voiceOrbCtx;
  const w = voiceOrbCanvas.width;
  const h = voiceOrbCanvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const orbR = Math.min(w, h) * 0.45;
  const t = nowMs * 0.001;
  
  const rotation = t * rotationSpeed;

  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = "rgba(2, 8, 20, 0.5)";
  ctx.fillRect(0, 0, w, h);

  ctx.globalCompositeOperation = "lighter";

  // Randomized hologram particles - no border, organic motion
  const particleCount = 32 + Math.floor(energy * 20);
  
  for (let i = 0; i < voiceOrbBlobs.length; i += 1) {
    const blob = voiceOrbBlobs[i];
    
    // Organic drift motion
    blob.drift += 0.04 + energy * 0.08;
    blob.angle += (blob.speed * 0.015) + (energy * 0.02);
    blob.orbit = 0.12 + Math.sin(t * 0.3 + i) * 0.35 + energy * 0.15;
    
    // Position with randomized paths
    const driftWave = Math.sin(blob.drift * 1.8) * 0.3 + Math.cos(blob.drift * 2.4) * 0.2;
    const orbitX = cx + Math.cos(blob.angle) * orbR * blob.orbit;
    const orbitY = cy + Math.sin(blob.angle) * orbR * blob.orbit;
    
    const px = orbitX + driftWave * orbR * 0.1;
    const py = orbitY + Math.sin(blob.drift) * orbR * 0.15;
    
    // 3D hologram glow balls
    const hue = (baseHue + blob.hueShift + t * 25 + Math.sin(blob.drift) * 40) % 360;
    const wobble = 0.08 + Math.sin(t * 1.5 + i) * 0.06;
    const particleRadius = orbR * (blob.radius + wobble);
    
    // Multi-layer glow for 3D depth
    const sizes = [
      { r: particleRadius * 3.5, a: glowIntensity * 0.12 * (0.5 + energy * 0.5) },
      { r: particleRadius * 2.2, a: glowIntensity * 0.25 * (0.6 + energy * 0.4) },
      { r: particleRadius * 1.3, a: glowIntensity * 0.45 * (0.7 + energy * 0.3) },
      { r: particleRadius * 0.6, a: glowIntensity * 0.8 * (0.8 + energy * 0.2) },
    ];
    
    sizes.forEach((layer) => {
      const grad = ctx.createRadialGradient(px, py, 0, px, py, layer.r);
      grad.addColorStop(0, `hsla(${hue}, 100%, 85%, ${layer.a * 1.3})`);
      grad.addColorStop(0.3, `hsla(${hue}, 100%, 65%, ${layer.a})`);
      grad.addColorStop(1, `hsla(${hue}, 100%, 45%, 0)`);
      
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, layer.r, 0, Math.PI * 2);
      ctx.fill();
    });
    
    // Bright core dot
    const coreGrad = ctx.createRadialGradient(px, py, 0, px, py, particleRadius);
    coreGrad.addColorStop(0, `hsla(${hue}, 100%, 95%, ${glowIntensity * 0.9 * (0.8 + energy * 0.2)})`);
    coreGrad.addColorStop(1, `hsla(${hue}, 100%, 70%, ${glowIntensity * 0.3})`);
    
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(px, py, particleRadius, 0, Math.PI * 2);
    ctx.fill();
  }

  // Central bright hologram core
  const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, orbR * 0.3);
  const coreHue = (baseHue + t * 35) % 360;
  const coreAlpha = glowIntensity * (0.5 + Math.sin(t * 2.2) * 0.3) * (0.6 + energy * 0.4);
  
  coreGrad.addColorStop(0, `hsla(${coreHue}, 100%, 95%, ${coreAlpha * 1.5})`);
  coreGrad.addColorStop(0.5, `hsla(${coreHue}, 100%, 75%, ${coreAlpha * 0.7})`);
  coreGrad.addColorStop(1, `hsla(${coreHue}, 100%, 55%, 0)`);
  
  ctx.fillStyle = coreGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, orbR * 0.28, 0, Math.PI * 2);
  ctx.fill();

  // Ultra bright center sparkle
  const centerGl = ctx.createRadialGradient(cx, cy, 0, cx, cy, orbR * 0.08);
  centerGl.addColorStop(0, `hsla(${coreHue}, 100%, 100%, ${coreAlpha * 2.0})`);
  centerGl.addColorStop(0.6, `hsla(${coreHue}, 100%, 85%, ${coreAlpha * 0.8})`);
  centerGl.addColorStop(1, "rgba(0, 0, 0, 0)");
  
  ctx.fillStyle = centerGl;
  ctx.beginPath();
  ctx.arc(cx, cy, orbR * 0.08, 0, Math.PI * 2);
  ctx.fill();

  voiceFrameId = requestAnimationFrame(animateVoiceParticles);
}

function updateUptime() {
  const elapsed = Math.floor((Date.now() - bootTime) / 1000);
  const hh = String(Math.floor(elapsed / 3600)).padStart(2, "0");
  const mm = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");
  if (uptimeChip) {
    uptimeChip.textContent = `UPTIME ${hh}:${mm}:${ss}`;
  }
}

function updateClock() {
  const now = new Date();
  const h = String(now.getHours()).padStart(2, "0");
  const m = String(now.getMinutes()).padStart(2, "0");
  const s = String(now.getSeconds()).padStart(2, "0");
  timeNow.textContent = `${h}:${m}:${s}`;

  dayName.textContent = now.toLocaleDateString("en-US", { weekday: "long" }).toUpperCase();
  monthDay.textContent = now.toLocaleDateString("en-US", { month: "long", day: "numeric" }).toUpperCase();
}

function seedBars(container) {
  if (!container) {
    return;
  }
  container.innerHTML = "";
  for (let i = 0; i < 18; i += 1) {
    const bar = document.createElement("i");
    bar.style.height = `${14 + Math.random() * 48}px`;
    container.appendChild(bar);
  }
}

function initBars() {
  seedBars(bars);
  seedBars(globalBars);
}

function animateBars() {
  [bars, globalBars].forEach((container, groupIdx) => {
    if (!container) {
      return;
    }
    const list = container.querySelectorAll("i");
    list.forEach((bar, idx) => {
      const phase = idx + groupIdx * 4;
      const base = 12 + ((Math.sin(Date.now() / 360 + phase) + 1) / 2) * 56;
      bar.style.height = `${base.toFixed(0)}px`;
    });
  });
  requestAnimationFrame(animateBars);
}

function pulseCoreIndex() {
  setInterval(() => {
    const val = 1 + Math.floor(Math.random() * 9);
    coreIndex.textContent = String(val);
    coreIndex.animate([
      { transform: "scale(1)", opacity: 1 },
      { transform: "scale(1.08)", opacity: 0.9 },
      { transform: "scale(1)", opacity: 1 },
    ], { duration: 420, easing: "ease-out" });
  }, 1800);
}

function simulateTelemetry() {
  let power = 100;
  setInterval(() => {
    power -= Math.random() * 0.22;
    if (power < 86) {
      power = 100;
    }
    powerPct.textContent = `${Math.round(power)}%`;
  }, 900);
}

function animateResources() {
  const nodes = Array.from(resourceList.querySelectorAll("li"));
  let p = 0;
  setInterval(() => {
    nodes.forEach((node, i) => {
      node.style.color = i === p ? "#d8fbff" : "#91d7f4";
      node.style.textShadow = i === p ? "0 0 14px #6cf6ff" : "none";
    });
    p = (p + 1) % nodes.length;
  }, 520);
}

function addLogRow(kind, text, meta = "") {
  if (!commandLog) {
    return;
  }
  const row = document.createElement("div");
  row.className = "log-row";

  const body = document.createElement("div");
  body.className = kind === "user" ? "log-user" : "log-ai";
  body.textContent = text;
  row.appendChild(body);

  if (meta) {
    const m = document.createElement("div");
    m.className = "log-meta";
    m.textContent = meta;
    row.appendChild(m);
  }

  commandLog.appendChild(row);
  commandLog.scrollTop = commandLog.scrollHeight;
}

// ...existing code...

function setMicState(text) {
  if (micStateChip) {
    micStateChip.textContent = `MIC: ${text}`;
  }
}

function setMode(nextMode) {
  inputMode = nextMode;
  if (modeTypeBtn) {
    modeTypeBtn.classList.toggle("active", nextMode === "type");
  }
  if (modeListenBtn) {
    modeListenBtn.classList.toggle("active", nextMode === "listen");
  }
  if (listenBtn) {
    listenBtn.classList.toggle("active", nextMode === "listen");
  }
  if (nextMode === "listen") {
    listenLoopEnabled = true;
    setMicState("ARMED");
    startListening(true);
  } else {
    listenLoopEnabled = false;
    recognitionRetryDelayMs = 260;
    stopListening();
  }
}

function isListenModeActive() {
  return inputMode === "listen" && listenLoopEnabled;
}

function armListenMode() {
  if (listenArmTimer) {
    window.clearInterval(listenArmTimer);
    listenArmTimer = null;
  }
  let attempts = 0;
  const maxAttempts = 12;
  startListening(true);
  listenArmTimer = window.setInterval(() => {
    if (!isListenModeActive() || recognitionActive || attempts >= maxAttempts) {
      window.clearInterval(listenArmTimer);
      listenArmTimer = null;
      return;
    }
    attempts += 1;
    startListening(true);
  }, 700);
}

function scheduleRecognitionRestart(delayMs = 260) {
  if (!isListenModeActive()) {
    return;
  }
  if (recognitionRestartTimer) {
    window.clearTimeout(recognitionRestartTimer);
  }
  recognitionRestartTimer = window.setTimeout(() => {
    recognitionRestartTimer = null;
    startListening();
  }, delayMs);
}

function stopListening() {
  if (listenArmTimer) {
    window.clearInterval(listenArmTimer);
    listenArmTimer = null;
  }
  if (recognitionRestartTimer) {
    window.clearTimeout(recognitionRestartTimer);
    recognitionRestartTimer = null;
  }
  if (!recognition) {
    setMicState("IDLE");
    return;
  }
  try {
    recognition.stop();
  } catch (_err) {
    // Ignore stop errors while recognizer is not active.
  }
  recognitionActive = false;
  setMicState("IDLE");
}

async function ensureMicReady(userInitiated = false) {
  if (micStream && micStream.active) {
    return true;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return true;
  }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    setupMicAnalyser(micStream);
    return true;
  } catch (err) {
    const now = Date.now();
    if (userInitiated || now - lastMicErrorLogAt > 5000) {
      addLogRow("ai", "Microphone permission required for listen mode.", String(err?.name || err || "mic-error"));
      lastMicErrorLogAt = now;
    }
    setMicState("BLOCKED");
    return false;
  }
}

function initVoices() {
  if (!("speechSynthesis" in window)) {
    return;
  }
  const voices = window.speechSynthesis.getVoices() || [];
  preferredVoice = voices.find((v) => /female|zira|samantha|aria|google us english/i.test(v.name)) || voices[0] || null;
}

function speakText(text) {
  if (!ttsEnabled || !("speechSynthesis" in window)) {
    return;
  }
  const content = (text || "").trim();
  if (!content) {
    return;
  }
  try {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(content);
    speakingActive = true;
    speechStartedAtMs = performance.now();
    if (preferredVoice) {
      utter.voice = preferredVoice;
    }
    utter.rate = 1.02;
    utter.pitch = 1.0;
    utter.volume = 1.0;
    utter.onend = () => {
      speakingActive = false;
      if (isListenModeActive()) {
        armListenMode();
      }
    };
    utter.onerror = () => {
      speakingActive = false;
      if (isListenModeActive()) {
        armListenMode();
      }
    };
    window.speechSynthesis.speak(utter);
  } catch (_err) {
    // Ignore TTS runtime errors silently.
  }
}

function stripWakePhrase(raw) {
  const text = (raw || "").trim();
  const lowered = text.toLowerCase();
  if (!lowered.startsWith(wakePhrase)) {
    return text || null;
  }
  const remainder = text.slice(wakePhrase.length).trim();
  return remainder || text;
}

function describeWeatherCode(code) {
  const map = {
    0: "CLEAR SKY",
    1: "MAINLY CLEAR",
    2: "PARTLY CLOUDY",
    3: "OVERCAST",
    45: "FOG",
    48: "RIME FOG",
    51: "LIGHT DRIZZLE",
    53: "DRIZZLE",
    55: "DENSE DRIZZLE",
    56: "FREEZING DRIZZLE",
    57: "DENSE FREEZING DRIZZLE",
    61: "LIGHT RAIN",
    63: "RAIN",
    65: "HEAVY RAIN",
    66: "FREEZING RAIN",
    67: "HEAVY FREEZING RAIN",
    71: "LIGHT SNOW",
    73: "SNOW",
    75: "HEAVY SNOW",
    77: "SNOW GRAINS",
    80: "RAIN SHOWERS",
    81: "HEAVY SHOWERS",
    82: "VIOLENT SHOWERS",
    85: "SNOW SHOWERS",
    86: "HEAVY SNOW SHOWERS",
    95: "THUNDERSTORM",
    96: "THUNDERSTORM HAIL",
    99: "SEVERE THUNDERSTORM",
  };
  return map[code] || "WEATHER LIVE";
}

function setWeatherDisplay(tempC, condition, locationText, forecastText = "") {
  if (typeof tempC === "number" && Number.isFinite(tempC) && tempVal) {
    tempVal.textContent = `${Math.round(tempC)}°`;
  }
  if (weatherLabel) {
    weatherLabel.textContent = condition || "WEATHER LIVE";
  }
  if (weatherLocation) {
    weatherLocation.textContent = locationText || "Live location unavailable";
  }
  if (weatherForecast) {
    weatherForecast.textContent = forecastText || "Forecast unavailable";
  }
}

function setWeatherLocationText(locationText) {
  lastWeatherLocationText = locationText || "Live location unavailable";
  if (weatherLocation) {
    weatherLocation.textContent = lastWeatherLocationText;
  }
}

function setClientLocationHint(payload) {
  if (!payload) {
    return;
  }
  const lat = Number(payload.lat);
  const lon = Number(payload.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return;
  }
  clientLocationHint = {
    lat,
    lon,
    place: String(payload.place || payload.location || "").trim(),
    source: String(payload.source || "").trim(),
    updatedAt: Date.now(),
  };
}

function setWeatherFallbackState(labelText, locationText, forecastText = "") {
  if (weatherLabel) {
    weatherLabel.textContent = labelText;
  }
  if (weatherLocation) {
    weatherLocation.textContent = locationText;
  }
  if (weatherForecast) {
    weatherForecast.textContent = forecastText;
  }
}

async function fetchJsonWithTimeout(url, timeoutMs = 4500, options = {}) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { ...options, signal: controller.signal });
    if (!resp.ok) {
      throw new Error(`http-${resp.status}`);
    }
    return await resp.json();
  } finally {
    window.clearTimeout(timer);
  }
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function getCurrentPositionPromise(options) {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) {
      reject(new Error("geolocation-not-supported"));
      return;
    }
    navigator.geolocation.getCurrentPosition(resolve, reject, options);
  });
}

async function getCurrentPositionWithHardTimeout(options, timeoutMs = 6500) {
  return Promise.race([
    getCurrentPositionPromise(options),
    sleep(timeoutMs).then(() => {
      throw new Error("geolocation-hard-timeout");
    }),
  ]);
}

async function reverseGeocode(lat, lon) {
  try {
    const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`;
    const data = await fetchJsonWithTimeout(url, 2800, { headers: { Accept: "application/json" } });
    const addr = data?.address || {};
    const city = addr.city || addr.town || addr.village || addr.hamlet || "";
    const state = addr.state || addr.region || "";
    if (city && state) {
      return `${city}, ${state}`;
    }
    return city || state || data?.display_name?.split(",")?.slice(0, 2)?.join(",") || "Your area";
  } catch (_err) {
    return "Your area";
  }
}

async function fetchWeatherByCoordinates(lat, lon, locationText = "") {
  const params = new URLSearchParams({
    latitude: String(lat),
    longitude: String(lon),
    current: "temperature_2m,weather_code",
    daily: "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
    forecast_days: "2",
    timezone: "auto",
  });
  const url = `https://api.open-meteo.com/v1/forecast?${params.toString()}`;
  const data = await fetchJsonWithTimeout(url, 4200);
  const current = data?.current || {};
  const daily = data?.daily || {};
  const temp = Number(current.temperature_2m);
  const condition = describeWeatherCode(Number(current.weather_code));
  const place = locationText || `${Number(lat).toFixed(2)}, ${Number(lon).toFixed(2)}`;
  const maxToday = Number((daily.temperature_2m_max || [])[0]);
  const minToday = Number((daily.temperature_2m_min || [])[0]);
  const rainProb = Number((daily.precipitation_probability_max || [])[0]);
  const tomorrowCode = Number((daily.weather_code || [])[1]);
  const tomorrowLabel = Number.isFinite(tomorrowCode) ? describeWeatherCode(tomorrowCode) : "N/A";
  const forecast = `Today H:${Number.isFinite(maxToday) ? Math.round(maxToday) : "--"}° L:${Number.isFinite(minToday) ? Math.round(minToday) : "--"}° Rain:${Number.isFinite(rainProb) ? Math.round(rainProb) : "--"}% | Tomorrow: ${tomorrowLabel}`;
  setWeatherDisplay(temp, condition, place, forecast);
  return {
    temp,
    condition,
    place,
    forecast,
    lat,
    lon,
    timestamp: Date.now(),
  };
}

function readCachedWeather() {
  try {
    const raw = window.localStorage.getItem(WEATHER_CACHE_KEY);
    if (!raw) {
      return null;
    }
    const payload = JSON.parse(raw);
    if (!payload || typeof payload !== "object") {
      return null;
    }
    return payload;
  } catch (_err) {
    return null;
  }
}

function writeCachedWeather(payload) {
  try {
    window.localStorage.setItem(WEATHER_CACHE_KEY, JSON.stringify(payload));
  } catch (_err) {
    // Ignore storage failures.
  }
}

function renderCityNews(city, headlines, options = {}) {
  if (cityNewsStatus) {
    const freshnessWindowHours = Number(options.freshnessWindowHours || 0);
    const isFallback = Boolean(options.isFallback);
    const freshnessTag = freshnessWindowHours > 0 ? `last ${freshnessWindowHours}h` : "latest";
    cityNewsStatus.textContent = `${city}: ${headlines.length} live updates (${freshnessTag}${isFallback ? ", fallback" : ""})`;
  }
  if (!cityNewsList) {
    return;
  }
  cityNewsList.innerHTML = "";
  if (!headlines.length) {
    const li = document.createElement("li");
    li.textContent = "No live local updates found in the last 8 hours.";
    cityNewsList.appendChild(li);
    return;
  }
  headlines.forEach((item) => {
    const li = document.createElement("li");
    const link = document.createElement("a");
    link.href = item.url;
    link.target = "_blank";
    link.rel = "noreferrer noopener";
    link.textContent = item.title;
    li.appendChild(link);

    const meta = document.createElement("div");
    meta.className = "news-meta";
    const parts = [item.source, item.age_label || item.published].filter(Boolean);
    meta.textContent = parts.join(" | ");
    li.appendChild(meta);
    cityNewsList.appendChild(li);
  });
}

async function fetchCityNews() {
  if (cityNewsStatus) {
    cityNewsStatus.textContent = "Fetching city updates...";
  }
  try {
    const city = String((clientLocationHint && clientLocationHint.place) || "").split(",")[0].trim();
    const url = city ? `/api/city-news?city=${encodeURIComponent(city)}` : "/api/city-news";
    const payload = await fetchJsonWithTimeout(url, 7000);
    if (payload?.ok) {
      renderCityNews(payload.city || "Your city", payload.headlines || [], {
        freshnessWindowHours: payload.freshness_window_hours,
        isFallback: payload.is_fallback,
      });
      return;
    }
  } catch (_err) {
    // Show fallback below.
  }
  if (cityNewsStatus) {
    cityNewsStatus.textContent = "City updates unavailable right now.";
  }
}

function renderGlobalNews(category, headlines) {
  if (!globalNewsList) {
    return;
  }
  globalNewsList.innerHTML = "";
  if (globalNewsStatus) {
    globalNewsStatus.textContent = `Global ${category} news • ${headlines.length} items`;
  }
  headlines.forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${item.title}</strong>`;
    if (item.summary) {
      const summary = document.createElement("p");
      summary.textContent = item.summary;
      summary.style.fontSize = "11px";
      summary.style.color = "#888";
      li.appendChild(summary);
    }
    const meta = document.createElement("div");
    meta.style.fontSize = "10px";
    meta.style.color = "#666";
    const parts = [];
    if (item.source) parts.push(item.source);
    if (item.published) parts.push(item.published);
    meta.textContent = parts.join(" | ");
    li.appendChild(meta);
    globalNewsList.appendChild(li);
  });
}

function formatMarketPrice(item) {
  const value = Number(item?.price);
  if (!Number.isFinite(value)) {
    return "--";
  }
  const currency = String(item?.currency || "").toUpperCase();
  if (currency === "INR") {
    return `INR ${value.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
  }
  if (currency === "USD") {
    return `$${value.toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
  }
  return `${value.toLocaleString("en-US", { maximumFractionDigits: 2 })} ${currency}`.trim();
}

function renderMarketPulse(items, updatedAt = "") {
  if (!marketList) {
    return;
  }
  marketList.innerHTML = "";
  if (!Array.isArray(items) || !items.length) {
    const li = document.createElement("li");
    li.textContent = "Market updates unavailable right now.";
    marketList.appendChild(li);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");

    const name = document.createElement("span");
    name.className = "market-item-name";
    name.textContent = item.name || item.symbol || "MARKET";
    li.appendChild(name);

    const valueWrap = document.createElement("span");
    valueWrap.className = "market-item-value";

    const price = document.createElement("span");
    price.className = "market-price";
    price.textContent = formatMarketPrice(item);
    valueWrap.appendChild(price);

    const change = document.createElement("span");
    const changePct = Number(item?.change_pct);
    change.className = "market-change";
    if (Number.isFinite(changePct)) {
      if (changePct > 0) {
        change.classList.add("up");
      } else if (changePct < 0) {
        change.classList.add("down");
      }
      change.textContent = `${changePct >= 0 ? "+" : ""}${changePct.toFixed(2)}%`;
    } else {
      change.textContent = "--";
    }
    valueWrap.appendChild(change);

    li.appendChild(valueWrap);
    marketList.appendChild(li);
  });

  if (marketStatus) {
    const stamp = updatedAt ? new Date(updatedAt).toLocaleTimeString("en-US", { hour12: false }) : "";
    marketStatus.textContent = stamp ? `Market pulse updated ${stamp}` : "Market pulse live";
  }
}

async function fetchMarketPulse() {
  const requestStartedAt = Date.now();
  if (marketStatus) {
    marketStatus.textContent = "Fetching market pulse...";
  }
  const pendingGuard = window.setTimeout(() => {
    if (marketStatus && marketStatus.textContent === "Fetching market pulse...") {
      marketStatus.textContent = "Market pulse unavailable right now.";
      renderMarketPulse([], "");
    }
  }, 8500);
  try {
    const payload = await fetchJsonWithTimeout("/api/market-pulse", 7000);
    if (payload?.ok) {
      renderMarketPulse(payload.items || [], payload.updated_at || "");
      window.clearTimeout(pendingGuard);
      return;
    }
  } catch (_err) {
    // Show fallback below.
  }
  window.clearTimeout(pendingGuard);
  if (marketStatus) {
    const elapsed = Date.now() - requestStartedAt;
    marketStatus.textContent = elapsed > 7000 ? "Market pulse timeout. Retrying shortly." : "Market pulse unavailable right now.";
  }
  renderMarketPulse([], "");
}

async function fetchGlobalNews(category = "") {
  if (globalNewsStatus) {
    globalNewsStatus.textContent = "Fetching global news...";
  }
  try {
    const url = category ? `/api/global-news?category=${encodeURIComponent(category)}` : "/api/global-news";
    const payload = await fetchJsonWithTimeout(url, 7000);
    if (payload?.ok) {
      renderGlobalNews(payload.category || "all", payload.headlines || []);
      return;
    }
  } catch (_err) {
    // Show fallback below.
  }
  if (globalNewsStatus) {
    globalNewsStatus.textContent = "Global news unavailable right now.";
  }
}

async function detectCoordinates() {
  try {
    const payload = await fetchJsonWithTimeout("/api/system-location", 3200);
    const lat = Number(payload?.lat);
    const lon = Number(payload?.lon);
    if (payload?.ok && Number.isFinite(lat) && Number.isFinite(lon)) {
      return {
        lat,
        lon,
        place: payload?.place || (payload?.source === "system" ? "System location" : "Approximate location"),
        source: payload?.source || "system",
      };
    }
  } catch (_backendErr) {
    // Continue to browser/IP fallbacks.
  }

  try {
    const pos = await getCurrentPositionWithHardTimeout(
      {
        enableHighAccuracy: false,
        timeout: 5000,
        maximumAge: 300000,
      },
      6500
    );
    return {
      lat: pos.coords.latitude,
      lon: pos.coords.longitude,
      place: "Live location",
      source: "gps",
    };
  } catch (_geoErr) {
    // Continue to IP fallback.
  }

  try {
    const ipData = await fetchJsonWithTimeout("https://ipwho.is/", 2800);
    if (ipData?.success && Number.isFinite(Number(ipData.latitude)) && Number.isFinite(Number(ipData.longitude))) {
      return {
        lat: Number(ipData.latitude),
        lon: Number(ipData.longitude),
        place: [ipData.city, ipData.region].filter(Boolean).join(", ") || "Approximate location",
        source: "ip",
      };
    }
  } catch (_ipErr) {
    // Continue to legacy IP fallback.
  }

  try {
    const data = await fetchJsonWithTimeout("https://ipapi.co/json/", 2800);
    const lat = Number(data?.latitude);
    const lon = Number(data?.longitude);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      return {
        lat,
        lon,
        place: [data?.city, data?.region].filter(Boolean).join(", ") || "Approximate location",
        source: "ip",
      };
    }
  } catch (_ipErr2) {
    // No location available.
  }

  return null;
}

async function fetchLiveWeather() {
  if (weatherRetryTimer) {
    window.clearTimeout(weatherRetryTimer);
    weatherRetryTimer = null;
  }
  setWeatherFallbackState("LOCATING WEATHER", lastWeatherLocationText, "Fetching live forecast...");

  try {
    const coords = await detectCoordinates();
    if (coords && Number.isFinite(Number(coords.lat)) && Number.isFinite(Number(coords.lon))) {
      setClientLocationHint(coords);
      const resolvedPlace = coords.place || await reverseGeocode(coords.lat, coords.lon);
      const weather = await fetchWeatherByCoordinates(coords.lat, coords.lon, resolvedPlace);
      setClientLocationHint({ ...coords, place: weather?.place || resolvedPlace });
      writeCachedWeather(weather);
      return;
    }
  } catch (_coordWeatherErr) {
    // Fall through to backend weather.
  }

  try {
    const payload = await fetchJsonWithTimeout("/api/system-weather", 5200);
    if (payload?.ok) {
      const temp = Number(payload?.temp_c);
      if (payload?.location) {
        setWeatherLocationText(payload.location);
      }
      setClientLocationHint({
        lat: payload?.lat,
        lon: payload?.lon,
        place: payload?.location || "System location",
        source: payload?.source || "system",
      });
      setWeatherDisplay(
        Number.isFinite(temp) ? temp : NaN,
        payload?.condition || "WEATHER LIVE",
        lastWeatherLocationText || payload?.location || "System location",
        payload?.forecast || "Forecast unavailable"
      );
      writeCachedWeather({
        temp: Number.isFinite(temp) ? temp : null,
        condition: payload?.condition || "WEATHER LIVE",
        place: payload?.location || "System location",
        forecast: payload?.forecast || "Forecast unavailable",
        timestamp: Date.now(),
      });
      return;
    }
  } catch (_backendWeatherErr) {
    // Use cache fallback below.
  }

  const cached = readCachedWeather();
  if (cached && Number.isFinite(Number(cached.temp))) {
    if (cached.place) {
      setWeatherLocationText(`${cached.place} (cached)`);
    }
    setWeatherDisplay(Number(cached.temp), cached.condition || "WEATHER LIVE", lastWeatherLocationText, cached.forecast || "Forecast unavailable");
    weatherRetryTimer = window.setTimeout(fetchLiveWeather, WEATHER_RETRY_MS);
    return;
  }

  setWeatherFallbackState("WEATHER UNAVAILABLE", lastWeatherLocationText || "Could not detect location", "Check connection and weather backend");
  weatherRetryTimer = window.setTimeout(fetchLiveWeather, WEATHER_RETRY_MS);
}

async function fetchSystemLocation() {
  try {
    const payload = await fetchJsonWithTimeout("/api/system-location", 3200);
    if (payload?.ok) {
      const label = payload.place || (payload.source === "system" ? `System location (${Number(payload.lat).toFixed(2)}, ${Number(payload.lon).toFixed(2)})` : "Approximate location");
      setClientLocationHint({ lat: payload.lat, lon: payload.lon, place: label, source: payload.source });
      setWeatherLocationText(label);
      return;
    }
  } catch (_err) {
    // Keep existing label if backend is temporarily slow.
  }
}

async function checkStatus() {
  if (!statusDot) {
    return;
  }
  try {
    const resp = await fetch("/api/status", { method: "GET", cache: "no-store" });
    if (!resp.ok) {
      throw new Error("status failed");
    }
    const data = await resp.json();
    statusDot.classList.add("online");
    if (Object.prototype.hasOwnProperty.call(data, "learning_enabled")) {
      learningUiState.enabled = Boolean(data.learning_enabled);
    }
    learningUiState.profile = data.learning_profile || learningUiState.profile;
    learningUiState.reasoningMode = data.reasoning_mode || learningUiState.reasoningMode;
    renderCommsLearning();
  } catch (_err) {
    statusDot.classList.remove("online");
    if (learningUiState.enabled === null) {
      learningUiState.enabled = false;
      learningUiState.reasoningMode = "offline";
      renderCommsLearning();
    }
  }
}

async function sendCommand(text) {
  lastUserInteractionAt = Date.now();
  addLogRow("user", `> ${text}`);
  const handledLocal = await handleVisualQuickCommand(text);
  if (handledLocal) {
    return;
  }
  try {
    const resp = await fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        location_hint: clientLocationHint
      }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      addLogRow("ai", data.response || "Command failed", data.error || "error");
      return;
    }
    const meta = `intent=${data.intent} confidence=${Number(data.confidence || 0).toFixed(2)}`;
    const responseText = data.response || "(no response)";
    addLogRow("ai", responseText, meta);
    if (Object.prototype.hasOwnProperty.call(data.metadata || {}, "learning_enabled")) {
      learningUiState.enabled = Boolean(data.metadata.learning_enabled);
    }
    learningUiState.lastIntent = data.intent || "unknown";
    learningUiState.lastConfidence = Number(data.confidence || 0);
    learningUiState.profile = data.metadata?.learning_profile || learningUiState.profile;
    learningUiState.reasoningMode = data.metadata?.reasoning_mode || learningUiState.reasoningMode;
    renderCommsLearning();
    
    // If user asked for news/updates, fetch appropriate news immediately
    const localNewsKeywords = /\b(news|city news|local news|updates|local updates|headlines|current events)\b/i;
    const globalNewsKeywords = /\b(global news|worldwide news|world news|trending|tech news|crypto news|business news)\b/i;
    
    if (globalNewsKeywords.test(text)) {
      let category = "";
      if (/tech/i.test(text)) category = "tech";
      else if (/crypto/i.test(text)) category = "crypto";
      else if (/business/i.test(text)) category = "business";
      setTimeout(() => fetchGlobalNews(category), 500);
    } else if (localNewsKeywords.test(text)) {
      setTimeout(fetchCityNews, 500);
    }
    
    speakText(responseText);
  } catch (err) {
    addLogRow("ai", "Backend unreachable", String(err));
  }
}

function ensureSpeechRecognition() {
  if (recognition) {
    return recognition;
  }
  const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Ctor) {
    addLogRow("ai", "Speech recognition not supported in this browser.", "Use TYPE mode.");
    return null;
  }

  recognition = new Ctor();
  recognition.lang = "en-US";
  recognition.continuous = true;
  recognition.interimResults = false;

  recognition.onstart = () => {
    recognitionActive = true;
    recognitionRetryDelayMs = 260;
    setMicState("LISTENING");
  };
  recognition.onerror = (event) => {
    recognitionActive = false;
    const errorCode = String(event.error || "unknown");
    setMicState("ERROR");
    addLogRow("ai", "Mic listen failed.", errorCode);
    const fatalError = errorCode === "not-allowed" || errorCode === "service-not-allowed";
    if (fatalError) {
      setMicState("PERMISSION");
      recognitionRetryDelayMs = Math.min(RECOGNITION_RETRY_MAX_MS, Math.max(recognitionRetryDelayMs, 1200));
      scheduleRecognitionRestart(recognitionRetryDelayMs);
      recognitionRetryDelayMs = Math.min(RECOGNITION_RETRY_MAX_MS, recognitionRetryDelayMs + 250);
      return;
    }
    if (isListenModeActive()) {
      recognitionRetryDelayMs = Math.min(RECOGNITION_RETRY_MAX_MS, recognitionRetryDelayMs + 120);
      scheduleRecognitionRestart(recognitionRetryDelayMs);
    }
  };
  recognition.onend = () => {
    recognitionActive = false;
    if (isListenModeActive()) {
      setMicState("RECONNECTING");
      scheduleRecognitionRestart(recognitionRetryDelayMs);
      recognitionRetryDelayMs = Math.min(RECOGNITION_RETRY_MAX_MS, recognitionRetryDelayMs + 80);
    } else {
      setMicState("IDLE");
    }
  };
  recognition.onresult = (event) => {
    const idx = typeof event.resultIndex === "number" ? event.resultIndex : (event.results?.length || 1) - 1;
    const transcript = event.results?.[idx]?.[0]?.transcript || "";
    if (!transcript) {
      return;
    }
    const parsed = stripWakePhrase(transcript);
    if (parsed) {
      sendCommand(parsed);
    }
  };

  return recognition;
}

async function startListening(userInitiated = false) {
  if (!isListenModeActive() && !userInitiated) {
    return;
  }
  const micReady = await ensureMicReady(userInitiated);
  if (!micReady) {
    return;
  }
  const rec = ensureSpeechRecognition();
  if (!rec) {
    return;
  }
  listenLoopEnabled = true;
  if (inputMode !== "listen") {
    inputMode = "listen";
  }
  if (recognitionActive) {
    setMicState("LISTENING");
    return;
  }
  try {
    rec.start();
  } catch (_err) {
    scheduleRecognitionRestart(300);
  }
}

if (commandForm && commandInput) {
  commandForm.addEventListener("submit", (event) => {
    event.preventDefault();
    lastUserInteractionAt = Date.now();
    const raw = commandInput.value.trim();
    const text = raw || (pendingCommandImageDataUrl ? "describe attached image" : "");
    if (!text) {
      return;
    }
    commandInput.value = "";
    const parsed = stripWakePhrase(text);
    if (parsed) {
      sendCommand(parsed);
    }
  });
}

// ...existing code...

if (modeTypeBtn) {
  modeTypeBtn.addEventListener("click", () => setMode("type"));
}

if (modeListenBtn) {
  modeListenBtn.addEventListener("click", () => {
    setMode("listen");
    armListenMode();
  });
}

if (listenBtn) {
  listenBtn.addEventListener("click", () => {
    setMode("listen");
    armListenMode();
  });
}

if (ttsToggleBtn) {
  ttsToggleBtn.addEventListener("click", () => {
    ttsEnabled = !ttsEnabled;
    ttsToggleBtn.classList.toggle("active", ttsEnabled);
    ttsToggleBtn.textContent = ttsEnabled ? "VOICE OUT: ON" : "VOICE OUT: OFF";
    if (!ttsEnabled && "speechSynthesis" in window) {
      speakingActive = false;
      window.speechSynthesis.cancel();
    }
  });
}

if (ttsStopBtn) {
  ttsStopBtn.addEventListener("click", () => {
    if ("speechSynthesis" in window) {
      speakingActive = false;
      window.speechSynthesis.cancel();
    }
  });
}

if (cameraToggleBtn) {
  cameraToggleBtn.addEventListener("click", () => {
    toggleCameraDetection();
  });
}

if (moodSummaryBtn) {
  moodSummaryBtn.addEventListener("click", () => {
    summarizeMoodHistory();
  });
}

window.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.code === "Space") {
    event.preventDefault();
    setMode("listen");
    armListenMode();
    return;
  }
  if (event.key === "Enter" && document.activeElement !== commandInput && inputMode === "type") {
    commandInput?.focus();
  }
});

window.setInterval(() => {
  if (!isListenModeActive()) {
    return;
  }
  if (!recognitionActive) {
    startListening(false);
  }
}, LISTEN_WATCHDOG_MS);

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && isListenModeActive()) {
    startListening(false);
  }
});

updateClock();
updateUptime();
renderCommsLearning();
initVoiceParticles();
animateVoiceParticles();

// Initialize camera detection
window.setTimeout(initCameraDetection, 1200);
updateCameraStatusUi();

setInterval(updateClock, 1000);
setInterval(updateUptime, 1000);
initBars();
animateBars();
pulseCoreIndex();
simulateTelemetry();
animateResources();
fetchSystemLocation();
fetchLiveWeather();
fetchCityNews();
fetchGlobalNews();
fetchMarketPulse();
window.setTimeout(fetchSystemLocation, 1500);
window.setTimeout(fetchLiveWeather, 2500);
window.setTimeout(fetchLiveWeather, 7000);
window.setTimeout(fetchCityNews, 2200);
window.setTimeout(fetchGlobalNews, 3500);
window.setTimeout(fetchMarketPulse, 1800);
setInterval(fetchSystemLocation, 5 * 60 * 1000);
setInterval(fetchLiveWeather, 10 * 60 * 1000);
setInterval(fetchCityNews, 5 * 60 * 1000);
setInterval(fetchGlobalNews, 8 * 60 * 1000);
setInterval(fetchMarketPulse, 2 * 60 * 1000);
checkStatus();
setInterval(checkStatus, 5000);
addLogRow("ai", "Jarvis web console ready.", "listen mode auto-start is enabled");
addLogRow("ai", "Wake phrase is optional.", "Type/say command directly, or use: Hey Jarvis <command>");
initVoices();
if ("speechSynthesis" in window) {
  window.speechSynthesis.onvoiceschanged = initVoices;
}

window.setTimeout(() => {
  setMode("listen");
  armListenMode();
}, 150);
