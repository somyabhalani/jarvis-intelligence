import json
import feedparser
import math
import platform
import random
import re
import subprocess
import sys
import threading
import os
from datetime import datetime, timezone
from datetime import timedelta
from email.utils import parsedate_to_datetime
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import urllib.error
from urllib.parse import parse_qs, quote_plus, urlparse
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jarvis as core


class JarvisRuntime:
    def vision_check(self, image_data_url: str, prompt: str = "", mode: str = "scene") -> Dict[str, Any]:
        """
        Calls NVIDIA Mistral Large 3 675B Instruct model via chat/completions API for image-to-text.
        """
        import requests
        api_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        api_key = os.getenv("NVIDIA_API_KEY", os.getenv("JARVIS_LLM_API_KEY", "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"))
        model = "meta/llama-3.2-11b-vision-instruct"
        try:
            print(f"[vision_check] Using model: {model}")
            print(f"[vision_check] Payload: prompt={prompt or 'Describe this image.'}, image_data_url_length={len(image_data_url)}")
            if not image_data_url.startswith("data:image/"):
                return {"ok": False, "error": "invalid image data url"}

            image_block = {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url,
                    "detail": "auto"
                }
            }
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Describe this image."},
                    image_block
                ]
            }
            payload = {
                "model": model,
                "messages": [user_message],
                "max_tokens": 2048,
                "temperature": 0.15,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": False
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json"
            }
            response = requests.post(api_url, headers=headers, json=payload, timeout=25)
            if response.status_code != 200:
                return {"ok": False, "error": f"NVIDIA API error: {response.status_code} {response.text}"}
            result = response.json()
            description = ""
            try:
                choices = result.get("choices", [])
                if choices:
                    description = choices[0]["message"].get("content", "").strip()
            except Exception:
                description = ""
            return {
                "ok": True,
                "description": description,
                "model": model,
                "raw": result,
            }
        except Exception as exc:
            return {"ok": False, "error": f"vision error: {exc}"}

    def _scrape_gold_price_inr(self):
        try:
            import requests
            from bs4 import BeautifulSoup
            url = "https://www.goodreturns.in/gold-rates/"
            resp = requests.get(url, timeout=8)
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("div", {"class": "gold_silver_table"})
            price_tag = table.find("td", text="24 carat").find_next_sibling("td")
            return float(price_tag.text.replace(",", ""))
        except Exception:
            return None

    def _scrape_silver_price_inr(self):
        try:
            import requests
            from bs4 import BeautifulSoup
            url = "https://www.goodreturns.in/silver-rates/"
            resp = requests.get(url, timeout=8)
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("div", {"class": "gold_silver_table"})
            price_tag = table.find("td", text="1 Kg").find_next_sibling("td")
            return float(price_tag.text.replace(",", ""))
        except Exception:
            return None

    def __init__(self) -> None:
        self.config = core.JarvisConfig.load()
        self.memory = core.MemoryStore(self.config.db_path)
        self.brain = core.AdvancedBrain(self.config, self.memory)
        self.lock = threading.Lock()
        self.latest_detection = {"mood": "neutral", "objects": [], "timestamp": ""}
        self.last_detection_context_signature = ""
        self.last_detection_context_at = 0.0
        self.casual_turn_count = 0
        self.detection_context_injections = 0

    def store_detection(self, detection_data: Dict[str, Any]) -> None:
        """Store detection data for context in responses."""
        with self.lock:
            payload = dict(detection_data or {})
            payload.setdefault("mood", "neutral")
            payload.setdefault("objects", [])
            payload.setdefault("timestamp", "")
            payload.setdefault("mood_confidence", 0.0)
            payload.setdefault("scene_confidence", 0.0)
            payload.setdefault("people_count", 0)
            payload.setdefault("face_count", 0)
            payload.setdefault("dominant_expression", "")
            payload.setdefault("object_detail", "")
            payload.setdefault("caption_text", "")
            payload.setdefault("frame_description", "")
            payload.setdefault("timeline", [])
            self.latest_detection = payload

    def get_detection_context(self) -> str:
        """Generate context string from latest detection for conversation integration."""
        # Keep this lightweight in casual chat: include visual context only early on.
        if self.detection_context_injections >= 1 or self.casual_turn_count > 6:
            return ""
        if not self.latest_detection:
            return ""
        objects = [str(x).strip() for x in (self.latest_detection.get("objects", []) or []) if str(x).strip()]
        mood = str(self.latest_detection.get("mood", "neutral") or "neutral").strip().lower()
        people_count = int(self.latest_detection.get("people_count", 0) or 0)
        scene_conf = float(self.latest_detection.get("scene_confidence", 0.0) or 0.0)
        mood_conf = float(self.latest_detection.get("mood_confidence", 0.0) or 0.0)
        expression = str(self.latest_detection.get("dominant_expression", "") or "").strip().lower()
        object_detail = str(self.latest_detection.get("object_detail", "") or "").strip()
        caption_text = str(self.latest_detection.get("caption_text", "") or "").strip()
        frame_description = str(self.latest_detection.get("frame_description", "") or "").strip()

        if scene_conf < 0.42 and mood_conf < 0.42 and not objects:
            return ""

        signature = f"{mood}|{people_count}|{'|'.join(objects[:5])}|{expression}"
        now = datetime.now(timezone.utc).timestamp()
        if signature == self.last_detection_context_signature and (now - self.last_detection_context_at) < 90:
            return ""

        people_text = ""
        if people_count > 1:
            people_text = f"I detect {people_count} people in frame. "
        elif people_count == 1:
            people_text = "I detect one person in frame. "

        object_text = ""
        if objects:
            object_text = f"Visible objects: {', '.join(objects[:5])}. "

        # Build the best available scene description.
        description_text = ""
        if frame_description:
            description_text = frame_description
        elif object_detail:
            description_text = object_detail
        elif caption_text:
            description_text = caption_text

        expression_text = f" Expression detected: {expression}." if expression and expression not in {"neutral", ""} else ""
        mood_text = f" Mood context: {mood}." if mood and mood != "neutral" else ""

        parts = []
        if people_text:
            parts.append(people_text.strip())
        if description_text:
            parts.append(description_text.strip())
        elif object_text:
            parts.append(object_text.strip())
        if expression_text:
            parts.append(expression_text.strip())
        if mood_text:
            parts.append(mood_text.strip())

        if not parts:
            return ""

        context = " ".join(parts)
        self.last_detection_context_signature = signature
        self.last_detection_context_at = now
        self.detection_context_injections += 1
        return f"[Vision context: {context}]"


    def execute(self, text: str, location_hint: Dict[str, Any] | None = None, image_data_url: str = "") -> Dict[str, Any]:
        cleaned = (text or "").strip()
        image_payload = str(image_data_url or "").strip()
        if not cleaned and not image_payload:
            return {
                "ok": False,
                "error": "empty command",
                "response": "Please enter a command.",
            }
        if not cleaned and image_payload.startswith("data:image/"):
            cleaned = "describe this image"

        hint = self._normalize_location_hint(location_hint)

        forced_visual_plain = re.sub(r"[^a-z0-9\s']", " ", cleaned.lower())
        forced_visual_plain = re.sub(r"\s+", " ", forced_visual_plain).strip().replace("'", "")
        force_hand_phrase = bool(
            re.search(r"\b(whats|what\s+is|what\s+s|what)\s+(?:in|on)\s+my\s+(?:hand|palm)\b", forced_visual_plain)
            or re.search(r"\bwhat\s+am\s+i\s+holding\b", forced_visual_plain)
        )

        asks_image_description = bool(
            re.search(r"\b(what(?:['’]?s| is)?\s+in\s+the\s+image|describe\s+(?:this\s+)?image|analy[sz]e\s+(?:this\s+)?image|what\s+do\s+you\s+see\s+in\s+(?:this\s+)?image)\b", cleaned.lower())
        )

        if asks_image_description and not image_payload.startswith("data:image/"):
            return {
                "ok": False,
                "intent": "system",
                "confidence": 0.2,
                "needs_confirmation": False,
                "response": "I did not receive an image payload. Please attach an image and try again.",
                "error": "missing image_data_url",
                "metadata": {
                    "action": "image_analysis_failed",
                    "reason": "missing_image",
                },
            }

        image_context = ""
        image_context_model = ""
        if image_payload.startswith("data:image/"):
            vision_mode = "hand" if (force_hand_phrase or self._is_visual_object_query(cleaned)) else "scene"
            vision_prompt = cleaned or ("what is in my hand" if vision_mode == "hand" else "describe this image")
            vision = self.vision_check(image_payload, vision_prompt, vision_mode)
            if vision.get("ok") and vision.get("description"):
                image_context = str(vision.get("description") or "").strip()
                image_context_model = str(vision.get("model") or "")
            else:
                return {
                    "ok": False,
                    "intent": "system",
                    "confidence": 0.2,
                    "needs_confirmation": False,
                    "response": "I received your image, but vision analysis failed right now. Please try again in a moment.",
                    "error": str(vision.get("error") or "vision unavailable"),
                    "metadata": {
                        "action": "image_analysis_failed",
                        "details": str(vision.get("details") or "")[:220],
                    },
                }

        if image_context and (asks_image_description or not cleaned):
            return {
                "ok": True,
                "intent": "system",
                "confidence": 0.95,
                "needs_confirmation": False,
                "response": image_context,
                "metadata": {
                    "action": "image_analysis",
                    "source": "uploaded_image",
                    "vision_model": image_context_model,
                },
            }

        if force_hand_phrase or self._is_visual_object_query(cleaned):
            if image_context:
                return {
                    "ok": True,
                    "intent": "system",
                    "confidence": 0.94,
                    "needs_confirmation": False,
                    "response": image_context,
                    "metadata": {
                        "action": "visual_object_check",
                        "source": "uploaded_image",
                        "vision_model": image_context_model,
                    },
                }

            latest = dict(self.latest_detection or {})
            detail = str(latest.get("frame_description") or latest.get("object_detail") or "").strip()
            caption = str(latest.get("caption_text") or "").strip()
            objects = [str(x).strip() for x in (latest.get("objects") or []) if str(x).strip()]
            people = int(latest.get("people_count") or 0)
            mood = str(latest.get("mood") or "neutral").strip().lower()

            if detail:
                response = detail
            elif caption:
                response = f"I checked the frame. {caption}."
            elif objects:
                response = f"I checked the frame and detected: {', '.join(objects[:6])}."
            else:
                response = (
                    "I tried to inspect the frame but do not have a confident object yet. "
                    "Keep the object closer to the camera, centered, and with better light, then ask again."
                )
            if people > 0:
                response += f" People in frame: {people}."
            response += f" Detected mood: {mood}."
            return {
                "ok": True,
                "intent": "system",
                "confidence": 0.91,
                "needs_confirmation": False,
                "response": response,
                "metadata": {
                    "action": "visual_object_check",
                    "objects": objects,
                    "people_count": people,
                    "mood": mood,
                },
            }

        if self._is_gpu_or_ip_query(cleaned):
            normalized = (cleaned or "").strip().lower()
            has_gpu = bool(re.search(r"\b(gpu|graphics|vram|video memory|gpu usage)\b", normalized))
            has_ip = bool(re.search(r"\b(ip address|public ip|local ip|my ip|internet ip|network ip|ip)\b", normalized))
            if has_gpu and has_ip:
                gpu = self.brain.system.get_gpu_report()
                ip = self.brain.system.get_ip_report()
                response = "\n\n".join([
                    str(gpu.get("report") or "GPU report unavailable."),
                    str(ip.get("report") or "IP report unavailable."),
                ])
                return {
                    "ok": bool(gpu.get("ok") or ip.get("ok")),
                    "intent": "system",
                    "confidence": 0.98,
                    "needs_confirmation": False,
                    "response": response,
                    "metadata": {"gpu": gpu, "ip": ip},
                }
            if has_gpu:
                gpu = self.brain.system.get_gpu_report()
                return {
                    "ok": bool(gpu.get("ok", False)),
                    "intent": "system",
                    "confidence": 0.97 if gpu.get("ok") else 0.65,
                    "needs_confirmation": False,
                    "response": str(gpu.get("report") or "GPU report unavailable."),
                    "metadata": gpu,
                }
            ip = self.brain.system.get_ip_report()
            return {
                "ok": bool(ip.get("ok", False)),
                "intent": "system",
                "confidence": 0.97 if ip.get("ok") else 0.65,
                "needs_confirmation": False,
                "response": str(ip.get("report") or "IP report unavailable."),
                "metadata": ip,
            }

        if self._is_traffic_query(cleaned):
            route = self._extract_route_query(cleaned)
            if route:
                origin, destination = route
                traffic = self.brain.system.get_route_traffic_report(origin, destination)
            else:
                city = self._extract_traffic_city(cleaned, hint if hint else None)
                traffic = self.brain.system.get_city_traffic_report(city)
            if traffic.get("report"):
                return {
                    "ok": bool(traffic.get("ok", True)),
                    "intent": "system",
                    "confidence": 0.95 if traffic.get("ok") else 0.75,
                    "needs_confirmation": False,
                    "response": str(traffic.get("report") or "Traffic opened in Google Maps."),
                    "metadata": traffic,
                }

        if self._is_weather_query(cleaned):
            weather = self.system_weather(hint if hint else None)
            if weather.get("ok"):
                return {
                    "ok": True,
                    "intent": "system",
                    "confidence": 0.98,
                    "needs_confirmation": False,
                    "response": weather.get("report") or "Weather is available.",
                    "metadata": {
                        "action": "weather",
                        "location": weather.get("location"),
                    },
                }

        # Keep location-sensitive news replies aligned with system location.
        if self._is_local_news_query(cleaned):
            city = ""
            if hint:
                city = self._extract_city_name(str(hint.get("place") or ""))
            payload = self.city_news(city)
            if payload.get("ok"):
                city = str(payload.get("city") or "your city").strip()
                headlines = payload.get("headlines") or []
                top = [str(item.get("title") or "").strip() for item in headlines[:3] if str(item.get("title") or "").strip()]
                if top:
                    response = f"Live local updates for {city}: " + " | ".join(top)
                else:
                    response = f"I am using your detected location and fetching local updates for {city}."
                return {
                    "ok": True,
                    "intent": "system",
                    "confidence": 0.95,
                    "needs_confirmation": False,
                    "response": response,
                    "metadata": {
                        "action": "fetch_city_news",
                        "city": city,
                    },
                }

        ingest_text = cleaned
        if image_context:
            ingest_text = f"{cleaned}\n\nAttached image context: {image_context}"

        with self.lock:
            plan = self.brain.ingest(ingest_text, source="web")
            self.casual_turn_count += 1

        metadata = dict(plan.metadata or {})
        metadata.setdefault("learning_profile", self.brain.user_model.profile_snapshot())
        metadata.setdefault("learning_enabled", bool(self.config.learning_enabled))
        metadata.setdefault("reasoning_mode", self.brain.reasoning_mode)
        if image_context:
            metadata["image_attached"] = True
            metadata["image_context"] = image_context
            metadata["image_model"] = image_context_model

        # Integrate detection context into response
        response_text = str(plan.response or "")
        detection_context = self.get_detection_context()
        if detection_context:
            response_text = f"{response_text} {detection_context}"

        return {
            "ok": True,
            "intent": plan.intent,
            "confidence": plan.confidence,
            "needs_confirmation": plan.needs_confirmation,
            "response": response_text,
            "metadata": metadata,
        }

    def _normalize_location_hint(self, hint: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(hint, dict):
            return {}
        lat = hint.get("lat")
        lon = hint.get("lon")
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            return {}
        if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
            return {}
        if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
            return {}
        return {
            "lat": lat_f,
            "lon": lon_f,
            "place": str(hint.get("place") or "").strip(),
            "source": str(hint.get("source") or "client").strip() or "client",
        }

    def _is_weather_query(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return bool(re.search(r"\b(weather|forecast|temperature|rain|humidity|wind|check weather)\b", normalized))

    def _is_visual_object_query(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        cleaned = re.sub(r"[^a-z0-9\s']", " ", normalized)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        plain = cleaned.replace("'", "")
        if not cleaned:
            return False

        # Hard override for common hand-object phrasings that should always trigger visual check.
        if re.search(r"\b(whats|what\s+is|what\s+s|what)\s+(?:in|on)\s+my\s+(?:hand|palm)\b", plain):
            return True
        if re.search(r"\bwhat\s+am\s+i\s+holding\b", plain):
            return True

        direct = bool(
            re.search(
                r"\b(what(?:['’]?s| is)?\s+(?:in|on)\s+my\s+(?:hand|palm)|what\s+am\s+i\s+holding|what\s+do\s+i\s+have\s+in\s+my\s+hand|what\s+is\s+in\s+my\s+palm|what\s+am\s+i\s+showing|what\s+did\s+i\s+show\s+you|what\s+is\s+this|can\s+you\s+identify\s+this|identify\s+(?:this\s+)?(?:item|object)|check\s+my\s+(?:hand|palm)|detect\s+(?:the\s+)?(?:item|object)|check\s+(?:the\s+)?(?:item|object)|look\s+at\s+my\s+(?:hand|palm)|see\s+my\s+(?:hand|palm)|tell\s+me\s+what\s+this\s+is|what\s+is\s+this\s+in\s+my\s+hand)\b",
                cleaned,
            )
        )
        if direct:
            return True

        has_intent = bool(re.search(r"\b(what|whats|which|identify|check|detect|tell|name|guess|recognize|recognise|find)\b", cleaned))
        has_context = bool(re.search(r"\b(hand|palm|holding|hold|showing|show|item|object|thing|this|it)\b", cleaned))
        return has_intent and has_context

    def _is_gpu_or_ip_query(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return bool(re.search(r"\b(gpu|graphics|vram|video memory|gpu usage|ip address|public ip|local ip|my ip|internet ip|network ip|\bip\b)\b", normalized))

    def _is_traffic_query(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return bool(re.search(r"\b(traffic|troffic|trafic|congestion|jam|route traffic|travel time|eta)\b", normalized))

    def _extract_route_query(self, text: str) -> Optional[Tuple[str, str]]:
        clean = (text or "").strip()
        if not clean:
            return None
        patterns = [
            r"\bfrom\s+(.+?)\s+to\s+(.+)$",
            r"\bbetween\s+(.+?)\s+and\s+(.+)$",
            r"\b(?:traffic|troffic|trafic)\s+(.+?)\s+to\s+(.+)$",
            r"^\s*(.+?)\s+to\s+(.+?)\s+(?:traffic|troffic|trafic)\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, clean, flags=re.I)
            if not match:
                continue
            origin = (match.group(1) or "").strip(" ,?.!")
            destination = (match.group(2) or "").strip(" ,?.!")
            origin = re.sub(r"\b(?:traffic|troffic|trafic|route|road|drive|driving|travel|eta|time)\b", "", origin, flags=re.I).strip(" ,?.!")
            destination = re.sub(r"\b(?:traffic|troffic|trafic|route|road|drive|driving|travel|eta|time|now|today|please)\b", "", destination, flags=re.I).strip(" ,?.!")
            if origin and destination and origin.lower() != destination.lower():
                return origin, destination
        return None

    def _extract_traffic_city(self, text: str, location_hint: Dict[str, Any] | None = None) -> Optional[str]:
        clean = (text or "").strip()
        if not clean:
            return self._extract_city_name(str((location_hint or {}).get("place") or "")) or None
        patterns = [
            r"\b(?:traffic|troffic|trafic|congestion|jam|route traffic|travel time|eta)\s+(?:in|at|for|of|near)\s+(.+)$",
            r"\b(?:how(?:'s| is)?\s+)?(?:traffic|troffic|trafic)\s+(.+)$",
            r"^\s*(.+?)\s+(?:traffic|troffic|trafic)\s*$",
        ]
        candidate = ""
        for pattern in patterns:
            match = re.search(pattern, clean, flags=re.I)
            if match:
                candidate = (match.group(1) or "").strip()
                if candidate:
                    break
        if not candidate:
            hinted = self._extract_city_name(str((location_hint or {}).get("place") or ""))
            return hinted or None
        candidate = re.sub(r"\b(?:today|now|right\s+now|currently|please)\b.*$", "", candidate, flags=re.I).strip(" ,?.!")
        candidate = re.sub(r"^(?:in|at|for|of|near)\s+", "", candidate, flags=re.I).strip()
        if not candidate:
            hinted = self._extract_city_name(str((location_hint or {}).get("place") or ""))
            return hinted or None
        return candidate

    def _is_local_news_query(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return bool(
            re.search(
                r"\b(local news|city news|news update|news updates|headlines|live updates|what.?s happening nearby|nearby news)\b",
                normalized,
            )
        )

    def status(self) -> Dict[str, Any]:
        models = self.brain.research.llm.pipeline_overview()
        profile = self.brain.user_model.profile_snapshot()
        return {
            "ok": True,
            "persona": self.config.persona_name,
            "reasoning_mode": self.brain.reasoning_mode,
            "talkative_level": self.brain.talkative_level,
            "language_mode": self.brain.conversation_language,
            "learning_enabled": bool(self.config.learning_enabled),
            "learning_profile": profile,
            "models": models,
        }

    def _windows_system_coordinates(self) -> Dict[str, Any]:
        script = (
            "Add-Type -AssemblyName System.Device; "
            "$w=New-Object System.Device.Location.GeoCoordinateWatcher; "
            "$null=$w.Start(); "
            "$deadline=(Get-Date).AddSeconds(6); "
            "while($w.Status -ne 'Ready' -and (Get-Date) -lt $deadline){ Start-Sleep -Milliseconds 300 }; "
            "$c=$w.Position.Location; "
            "if($c -and -not $c.IsUnknown){ \"$($c.Latitude),$($c.Longitude)\" }"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        out = (completed.stdout or "").strip()
        if not out or "," not in out:
            raise RuntimeError("windows location unavailable")
        lat_str, lon_str = out.split(",", 1)
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
        return {"lat": lat, "lon": lon, "source": "system"}

    def _ip_coordinates(self) -> Dict[str, Any]:
        candidates = [
            ("https://ipwho.is/", "ipwho"),
            ("https://ip-api.com/json/", "ipapi2"),
            ("https://ipinfo.io/json", "ipinfo"),
            ("https://ipapi.co/json/", "ipapi"),
            ("https://geolocation-db.com/json/", "geodb"),
        ]
        for url, kind in candidates:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=4) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                if kind == "ipwho":
                    if data.get("success"):
                        lat = float(data.get("latitude"))
                        lon = float(data.get("longitude"))
                        place = ", ".join([part for part in [data.get("city"), data.get("region")] if part]) or "Approximate location"
                        return {"lat": lat, "lon": lon, "source": "ip", "place": place}
                if kind == "ipapi2":
                    if data.get("status") == "success":
                        lat = float(data.get("lat"))
                        lon = float(data.get("lon"))
                        place = ", ".join([part for part in [data.get("city"), data.get("regionName")] if part]) or "Approximate location"
                        return {"lat": lat, "lon": lon, "source": "ip", "place": place}
                elif kind == "ipinfo":
                    loc = str(data.get("loc") or "")
                    if "," in loc:
                        lat_str, lon_str = loc.split(",", 1)
                        lat = float(lat_str)
                        lon = float(lon_str)
                        place = ", ".join([part for part in [data.get("city"), data.get("region")] if part]) or "Approximate location"
                        return {"lat": lat, "lon": lon, "source": "ip", "place": place}
                elif kind == "ipapi":
                    lat = float(data.get("latitude"))
                    lon = float(data.get("longitude"))
                    place = ", ".join([part for part in [data.get("city"), data.get("region")] if part]) or "Approximate location"
                    return {"lat": lat, "lon": lon, "source": "ip", "place": place}
                elif kind == "geodb":
                    lat = float(data.get("latitude"))
                    lon = float(data.get("longitude"))
                    city = data.get("city") or ""
                    country = data.get("country_name") or ""
                    place = ", ".join([part for part in [city, country] if part]) or "Approximate location"
                    return {"lat": lat, "lon": lon, "source": "ip", "place": place}
            except Exception:
                continue
        raise RuntimeError("ip location unavailable")

    def system_location(self) -> Dict[str, Any]:
        try:
            result = self._ip_coordinates()
            return {"ok": True, **result}
        except Exception as exc:
            if platform.system().lower().startswith("win"):
                try:
                    result = self._windows_system_coordinates()
                    return {"ok": True, **result}
                except Exception:
                    pass
            return {"ok": False, "error": f"location unavailable: {exc}"}

    def system_weather(self, location_override: Dict[str, Any] | None = None) -> Dict[str, Any]:
        loc = location_override if isinstance(location_override, dict) and location_override else self.system_location()
        location_input = None
        if loc.get("ok") and loc.get("lat") is not None and loc.get("lon") is not None:
            location_input = f"{loc['lat']},{loc['lon']}"
        elif loc.get("lat") is not None and loc.get("lon") is not None:
            location_input = f"{loc['lat']},{loc['lon']}"

        weather = self.brain.system.get_weather_report(location_input)
        if not weather.get("ok"):
            return {
                "ok": False,
                "error": weather.get("error", "weather unavailable"),
                "location": loc,
            }

        report = str(weather.get("report", ""))
        lines = [line.strip() for line in report.splitlines() if line.strip()]
        condition = "WEATHER LIVE"
        temp_c = None
        forecast_today = ""
        forecast_tomorrow = ""

        for line in lines:
            lower = line.lower()
            if lower.startswith("condition:"):
                condition = line.split(":", 1)[1].strip() or condition
            elif lower.startswith("temperature:"):
                match = re.search(r"(-?\d+)\s*°?c", line, flags=re.I)
                if match:
                    temp_c = int(match.group(1))
            elif lower.startswith("forecast today:"):
                forecast_today = line.split(":", 1)[1].strip()
            elif lower.startswith("forecast tomorrow:"):
                forecast_tomorrow = line.split(":", 1)[1].strip()

        forecast = ""
        if forecast_today and forecast_tomorrow:
            forecast = f"Today: {forecast_today} | Tomorrow: {forecast_tomorrow}"
        elif forecast_today:
            forecast = f"Today: {forecast_today}"
        elif forecast_tomorrow:
            forecast = f"Tomorrow: {forecast_tomorrow}"

        location_text = weather.get("location") or "System location"
        if isinstance(loc, dict) and loc.get("ok") and loc.get("source") == "system":
            location_text = f"{location_text} (system)"
        elif isinstance(loc, dict) and loc.get("source") == "client" and loc.get("place"):
            location_text = str(loc.get("place"))

        return {
            "ok": True,
            "source": "system",
            "lat": loc.get("lat") if isinstance(loc, dict) else None,
            "lon": loc.get("lon") if isinstance(loc, dict) else None,
            "location": location_text,
            "condition": condition,
            "temp_c": temp_c,
            "forecast": forecast,
            "report": report,
        }

    def _extract_city_name(self, text: str) -> str:
        clean = (text or "").replace("(system)", "").replace("(cached)", "").strip()
        if not clean:
            return ""
        return clean.split(",", 1)[0].strip()

    def _infer_city_for_news(self) -> str:
        weather = self.system_weather()
        if weather.get("ok"):
            city = self._extract_city_name(str(weather.get("location", "")))
            if city:
                return city
        loc = self.system_location()
        if loc.get("ok"):
            city = self._extract_city_name(str(loc.get("place", "")))
            if city:
                return city
        return "India"

    _city_news_cache = {}
    _city_news_lock = threading.Lock()

    # Mapping of city names to Times of India RSS feed codes
    _toi_city_rss = {
        "mumbai": "-2128838584",
        "delhi": "-2128838594",
        "bangalore": "-2128833038",
        "bengaluru": "-2128833038",
        "hyderabad": "-2128816011",
        "ahmedabad": "-2128838215",
        "chennai": "-2128834400",
        "kolkata": "-2128830821",
        "pune": "-2128821991",
        "jaipur": "-2128837754",
        "lucknow": "-2128837163",
        "kanpur": "-2128843683",
        "nagpur": "-2128838597",
        "vadodara": "-2128838597",
        "surat": "-2128821153",
        "patna": "-2128821738",
        "bhopal": "-2128817663",
        "indore": "-2128831681",
        "chandigarh": "-2128816762",
        "agra": "-2128815552",
        "varanasi": "-2128820891",
        "rajkot": "-2128823662",
        "ranchi": "-2128825702",
        "coimbatore": "-2128822721",
        "thiruvananthapuram": "-2128830415",
        "kochi": "-2128830192",
        "bhubaneswar": "-2128819673",
        "visakhapatnam": "-2128820471",
        "aurangabad": "-2128820938",
        "meerut": "-2128820381",
        "goa": "-3012537560",
        # Add more as needed
    }


    def _get_google_news_rss_url(self, city: str, state: str = "") -> str:
        # Compose a Google News RSS query for city and state
        query = city
        if state and state.lower() not in city.lower():
            query += f" {state}"
        query = query.replace(" ", "+")
        return f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    def _refresh_city_news(self, city: str, state: str = ""):
        now_utc = datetime.now(timezone.utc)
        rss_url = self._get_google_news_rss_url(city, state)
        all_headlines = []
        filtered_headlines = []
        city_lower = city.lower()
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:16]:
                published = entry.get("published", "")
                published_dt = now_utc
                try:
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    published_dt = now_utc
                age = now_utc - published_dt
                age_minutes = int(age.total_seconds() // 60)
                if age_minutes < 60:
                    age_label = f"{age_minutes}m ago"
                else:
                    age_label = f"{max(1, age_minutes // 60)}h ago"
                headline = {
                    "title": entry.title,
                    "url": entry.link,
                    "source": entry.get("source", {}).get("title", "Google News"),
                    "published": published,
                    "age_minutes": age_minutes,
                    "age_label": age_label,
                    "summary": getattr(entry, "summary", "")
                }
                all_headlines.append(headline)
                # Strict filter: only include if city name is in title or summary
                if city_lower in entry.title.lower() or city_lower in getattr(entry, "summary", "").lower():
                    filtered_headlines.append(headline)
        except Exception as exc:
            all_headlines = []
            filtered_headlines = []
        # Only show filtered headlines; if none, show a message
        if filtered_headlines:
            headlines = filtered_headlines[:8]
        else:
            headlines = [{
                "title": f"No recent local news found for {city}.",
                "url": "",
                "source": "Google News",
                "published": "",
                "age_minutes": 0,
                "age_label": "",
                "summary": ""
            }]
        with self._city_news_lock:
            self._city_news_cache[city_lower] = {
                "headlines": headlines,
                "updated_at": now_utc,
            }

    def _ensure_city_news_fresh(self, city: str, state: str = ""):
        now_utc = datetime.now(timezone.utc)
        with self._city_news_lock:
            cache = self._city_news_cache.get(city.lower())
            updated_at = cache["updated_at"] if cache else None
            if not updated_at or (now_utc - updated_at).total_seconds() > 60:
                threading.Thread(target=self._refresh_city_news, args=(city, state), daemon=True).start()

    def city_news(self, city: str = "", state: str = "") -> Dict[str, Any]:
        # If city not provided, infer it
        if not city:
            city = self._infer_city_for_news()
        # Try to infer state if not provided (from weather/location if available)
        if not state:
            # Try to get state from weather/location (very basic, can be improved)
            weather = self.system_weather()
            state = ""
            if weather.get("ok"):
                loc = str(weather.get("location", ""))
                if "," in loc:
                    parts = [p.strip() for p in loc.split(",")]
                    if len(parts) > 1:
                        state = parts[-1]
        self._ensure_city_news_fresh(city, state)
        with self._city_news_lock:
            cache = self._city_news_cache.get(city.lower(), {})
            headlines = cache.get("headlines", [])
            updated_at = cache.get("updated_at")
        return {
            "ok": True,
            "city": city,
            "state": state,
            "headlines": headlines,
            "provider": "google_news",
            "updated_at": updated_at.isoformat() if updated_at else None,
        }

    def global_news(self, category: str = "") -> Dict[str, Any]:
        """Fetch global news from multiple sources."""
        sources = {
            "all": [
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://feeds.reuters.com/reuters/topNews",
                "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
            ],
            "tech": [
                "https://feeds.techcrunch.com/",
                "https://news.ycombinator.com/rss",
            ],
            "crypto": [
                "https://feeds.coindesk.com/latest",
            ],
            "business": [
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://feeds.reuters.com/finance",
            ],
        }
        target_category = (category or "all").lower().strip() or "all"
        feed_urls = sources.get(target_category, sources["all"])
        
        for url in feed_urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=6) as resp:
                    raw = resp.read()
                root = ET.fromstring(raw)
                headlines = []
                for item in root.findall("./channel/item")[:10]:
                    title = (item.findtext("title") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    pub_date = (item.findtext("pubDate") or "").strip()
                    desc = (item.findtext("description") or "").strip()
                    if not title or not link:
                        continue
                    source = ""
                    if " - " in title:
                        source = title.rsplit(" - ", 1)[-1].strip()
                    headlines.append({
                        "title": title,
                        "url": link,
                        "source": source,
                        "published": pub_date,
                        "summary": desc[:200] if desc else "",
                    })
                if headlines:
                    return {"ok": True, "category": target_category, "headlines": headlines}
            except Exception:
                continue
        
        return {"ok": False, "error": "global news unavailable at this time", "category": target_category}

    def market_pulse(self) -> Dict[str, Any]:
        items = []
        errors = []

        try:
            crypto_url = (
                "https://api.coingecko.com/api/v3/simple/price"
                "?ids=bitcoin,ethereum&vs_currencies=inr"
                "&include_24hr_change=true"
            )
            req = urllib.request.Request(crypto_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))

            crypto_map = [
                ("bitcoin", "BTC", "Bitcoin"),
                ("ethereum", "ETH", "Ethereum"),
            ]
            for key, symbol, name in crypto_map:
                row = data.get(key) or {}
                price = row.get("inr")
                change = row.get("inr_24h_change")
                if isinstance(price, (int, float)):
                    items.append({
                        "kind": "crypto",
                        "symbol": symbol,
                        "name": name,
                        "currency": "INR",
                        "price": float(price),
                        "change_pct": float(change) if isinstance(change, (int, float)) else None,
                    })
        except Exception as exc:
            errors.append(f"crypto: {exc}")

        try:
            quote_url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=%5ENSEI,%5EBSESN"
            req = urllib.request.Request(quote_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))

            results = ((data.get("quoteResponse") or {}).get("result") or [])
            label_map = {
                "^NSEI": "NIFTY 50",
                "^BSESN": "SENSEX",
            }
            for row in results:
                symbol = str(row.get("symbol") or "").strip()
                if symbol not in label_map:
                    continue
                price = row.get("regularMarketPrice")
                change_pct = row.get("regularMarketChangePercent")
                if isinstance(price, (int, float)):
                    items.append({
                        "kind": "index",
                        "symbol": symbol,
                        "name": label_map[symbol],
                        "currency": "INR",
                        "price": float(price),
                        "change_pct": float(change_pct) if isinstance(change_pct, (int, float)) else None,
                    })
        except Exception as exc:
            errors.append(f"indices: {exc}")

        # Yahoo quote endpoint occasionally omits index rows. Fall back to chart API.
        existing_symbols = {str(item.get("symbol") or "") for item in items}
        for symbol, label in [("^NSEI", "NIFTY 50"), ("^BSESN", "SENSEX")]:
            if symbol in existing_symbols:
                continue
            try:
                encoded = quote_plus(symbol)
                chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?interval=1d&range=5d"
                req = urllib.request.Request(chart_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=6) as resp:
                    chart_data = json.loads(resp.read().decode("utf-8", errors="ignore"))

                result = (((chart_data.get("chart") or {}).get("result") or [None])[0] or {})
                meta = result.get("meta") or {}
                close_series = (((result.get("indicators") or {}).get("quote") or [{}])[0] or {}).get("close") or []
                closes = [float(x) for x in close_series if isinstance(x, (int, float))]
                if not closes:
                    continue

                last_price = closes[-1]
                prev_close = meta.get("chartPreviousClose")
                if not isinstance(prev_close, (int, float)):
                    prev_close = closes[-2] if len(closes) >= 2 else None
                change_pct = None
                if isinstance(prev_close, (int, float)) and prev_close:
                    change_pct = ((last_price - float(prev_close)) / float(prev_close)) * 100

                items.append({
                    "kind": "index",
                    "symbol": symbol,
                    "name": label,
                    "currency": str(meta.get("currency") or "INR").upper(),
                    "price": float(last_price),
                    "change_pct": float(change_pct) if isinstance(change_pct, (int, float)) else None,
                })
            except Exception as exc:
                errors.append(f"{symbol} chart: {exc}")

        if not items:
            return {
                "ok": False,
                "error": "market pulse unavailable",
                "details": errors,
            }

        return {
            "ok": True,
            "items": items,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
        }


RUNTIME = JarvisRuntime()


class JarvisHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._send_json(RUNTIME.status())
            return
        if path == "/api/system-location":
            payload = RUNTIME.system_location()
            status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(payload, status=status)
            return
        if path == "/api/system-weather":
            payload = RUNTIME.system_weather()
            status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(payload, status=status)
            return
        if path == "/api/city-news":
            query = urlparse(self.path).query
            params = parse_qs(query)
            city = (params.get("city", [""])[0])
            payload = RUNTIME.city_news(city)
            status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(payload, status=status)
            return
        if path == "/api/global-news":
            query = urlparse(self.path).query
            params = parse_qs(query)
            category = (params.get("category", [""])[0])
            payload = RUNTIME.global_news(category)
            status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(payload, status=status)
            return
        if path == "/api/market-pulse":
            payload = RUNTIME.market_pulse()
            status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(payload, status=status)
            return
        if path == "/api/health":
            self._send_json({"ok": True, "service": "jarvis-webapp"})
            return
        return super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json({"ok": False, "error": "invalid json"}, status=HTTPStatus.BAD_REQUEST)
            return

        if path == "/api/detection":
            RUNTIME.store_detection(data)
            self._send_json({"ok": True})
            return

        if path == "/api/vision-check":
            image_data_url = str(data.get("image_data_url", "")) if isinstance(data, dict) else ""
            prompt = str(data.get("prompt", "")) if isinstance(data, dict) else ""
            mode = str(data.get("mode", "hand")) if isinstance(data, dict) else "hand"
            if not image_data_url:
                self._send_json({"ok": False, "error": "image_data_url is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            payload = RUNTIME.vision_check(image_data_url, prompt, mode)
            status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(payload, status=status)
            return
        
        if path != "/api/command":
            self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return

        text = str(data.get("text", ""))
        location_hint = data.get("location_hint") if isinstance(data, dict) else None
        image_data_url = str(data.get("image_data_url", "")) if isinstance(data, dict) else ""
        result = RUNTIME.execute(
            text,
            location_hint if isinstance(location_hint, dict) else None,
            image_data_url,
        )
        status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
        self._send_json(result, status=status)


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8080))
    server = ThreadingHTTPServer((host, port), JarvisHandler)
    print(f"Jarvis webapp running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
