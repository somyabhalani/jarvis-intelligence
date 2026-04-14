"""Jarvis core foundation.

A compact, self-contained Jarvis-style assistant with:
- NVIDIA/DeepSeek default LLM access
- research and system tools
- memory and light personalization
- voice mode with wake phrase / hotkey
- a dashboard-style animated UI
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import difflib
import html
import json
import math
import os
import platform
import queue
import random
import re
import requests
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import textwrap
import tempfile
import time
import urllib.parse
import urllib.request
import webbrowser
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

# Vision feature imports removed - focus on core intelligence
# (GestureControl, FaceAuth, EmotionDetector deprecated)

APP_NAME = "Jarvis"
DEFAULT_DB_PATH = Path.home() / ".jarvis" / "jarvis.db"
DEFAULT_CONFIG_PATH = Path.home() / ".jarvis" / "config.json"
MAX_SESSION_TURNS = 20
DEFAULT_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_LLM_API_KEY = os.getenv("NVIDIA_API_KEY", os.getenv("JARVIS_LLM_API_KEY", "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"))
DEFAULT_LLM_MODEL = "deepseek-ai/deepseek-v3.1-terminus"
DEFAULT_FAST_MODEL = os.getenv("JARVIS_FAST_MODEL", "meta/llama-3.1-8b-instruct")
DEFAULT_INTENT_MODEL = os.getenv("JARVIS_INTENT_MODEL", "meta/llama-3.1-8b-instruct")
DEFAULT_DEEP_MODEL = os.getenv("JARVIS_DEEP_MODEL", DEFAULT_LLM_MODEL)
DEFAULT_PLANNING_MODEL = os.getenv("JARVIS_PLANNING_MODEL", "meta/llama-3.1-70b-instruct")
DEFAULT_CODER_MODEL = os.getenv("JARVIS_CODER_MODEL", "deepseek-ai/deepseek-v3.1-terminus")
DEFAULT_REASONING_MODEL = os.getenv("JARVIS_REASONING_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")
DEFAULT_LONG_CONTEXT_MODEL = os.getenv("JARVIS_LONG_CONTEXT_MODEL", "meta/llama-3.1-405b-instruct")
DEFAULT_VOICE_HOTKEY = "ctrl+space"


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text.strip().lower())


def safe_json_loads(raw: str, fallback: Any) -> Any:
	try:
		return json.loads(raw)
	except Exception:
		return fallback


def clamp(value: float, low: float, high: float) -> float:
	return max(low, min(high, value))


def tokenise(text: str) -> List[str]:
	return re.findall(r"[a-zA-Z0-9']+", text.lower())


def html_unescape(text: str) -> str:
	return html.unescape(text)


@dataclass
class JarvisConfig:
	"""Runtime configuration for the assistant."""

	db_path: Path = DEFAULT_DB_PATH
	config_path: Path = DEFAULT_CONFIG_PATH
	persona_name: str = APP_NAME
	assistant_tone: str = "calm, sharp, witty, detailed"
	allow_proactive: bool = True
	memory_window: int = MAX_SESSION_TURNS
	max_search_results: int = 5
	confidence_threshold: float = 0.55
	learning_enabled: bool = True
	store_sensitive_memory: bool = False
	preferred_response_length: str = "detailed"
	language: str = "en"
	# Vision features removed - focus on task intelligence

	@classmethod
	def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "JarvisConfig":
		if not path.exists():
			return cls(config_path=path)
		data = safe_json_loads(path.read_text(encoding="utf-8"), {})
		return cls(
			db_path=Path(data.get("db_path", str(DEFAULT_DB_PATH))),
			config_path=path,
			persona_name=data.get("persona_name", APP_NAME),
			assistant_tone=data.get("assistant_tone", "calm, sharp, witty, detailed"),
			allow_proactive=bool(data.get("allow_proactive", True)),
			memory_window=int(data.get("memory_window", MAX_SESSION_TURNS)),
			max_search_results=int(data.get("max_search_results", 5)),
			confidence_threshold=float(data.get("confidence_threshold", 0.55)),
			learning_enabled=bool(data.get("learning_enabled", True)),
			store_sensitive_memory=bool(data.get("store_sensitive_memory", False)),
			preferred_response_length=data.get("preferred_response_length", "detailed"),
			language=data.get("language", "en"),

		)

	def save(self) -> None:
		ensure_parent(self.config_path)
		payload = dataclasses.asdict(self)
		payload["db_path"] = str(self.db_path)
		payload["config_path"] = str(self.config_path)

		self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class ConversationTurn:
	role: str
	content: str
	timestamp: str = field(default_factory=utc_now_iso)
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSignal:
	text_style: str = "neutral"
	urgency: float = 0.0
	stress: float = 0.0
	focus: float = 0.5
	formality: float = 0.5
	humor_preference: float = 0.5
	task_intensity: float = 0.5
	mood: str = "neutral"
	source: str = "text"


@dataclass
class ActionPlan:
	intent: str
	response: str
	confidence: float
	needs_confirmation: bool = False
	tool_calls: List[Dict[str, Any]] = field(default_factory=list)
	memory_updates: List[Dict[str, Any]] = field(default_factory=list)
	metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
	"""SQLite-based memory and lightweight learning storage."""

	def __init__(self, db_path: Path):
		self.db_path = db_path
		ensure_parent(self.db_path)
		self._init_db()

	def _connect(self) -> sqlite3.Connection:
		conn = sqlite3.connect(self.db_path)
		conn.row_factory = sqlite3.Row
		return conn

	def _init_db(self) -> None:
		with self._connect() as conn:
			conn.executescript(
				"""
				CREATE TABLE IF NOT EXISTS turns (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					role TEXT NOT NULL,
					content TEXT NOT NULL,
					timestamp TEXT NOT NULL,
					metadata TEXT NOT NULL DEFAULT '{}'
				);

				CREATE TABLE IF NOT EXISTS preferences (
					key TEXT PRIMARY KEY,
					value TEXT NOT NULL,
					confidence REAL NOT NULL DEFAULT 0.5,
					updated_at TEXT NOT NULL
				);

				CREATE TABLE IF NOT EXISTS facts (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					category TEXT NOT NULL,
					key TEXT NOT NULL,
					value TEXT NOT NULL,
					confidence REAL NOT NULL DEFAULT 0.5,
					source TEXT NOT NULL DEFAULT 'interaction',
					created_at TEXT NOT NULL,
					updated_at TEXT NOT NULL,
					UNIQUE(category, key)
				);

				CREATE TABLE IF NOT EXISTS learning_events (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					event_type TEXT NOT NULL,
					payload TEXT NOT NULL,
					timestamp TEXT NOT NULL
				);

				CREATE TABLE IF NOT EXISTS reminders (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					title TEXT NOT NULL,
					due_at TEXT,
					status TEXT NOT NULL DEFAULT 'pending',
					metadata TEXT NOT NULL DEFAULT '{}',
					created_at TEXT NOT NULL,
					updated_at TEXT NOT NULL
				);

				CREATE TABLE IF NOT EXISTS audit_log (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					action TEXT NOT NULL,
					detail TEXT NOT NULL,
					success INTEGER NOT NULL DEFAULT 1,
					timestamp TEXT NOT NULL
				);

				CREATE TABLE IF NOT EXISTS mood_samples (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					mood TEXT NOT NULL,
					stress REAL NOT NULL DEFAULT 0.0,
					focus REAL NOT NULL DEFAULT 0.5,
					source TEXT NOT NULL DEFAULT 'text',
					timestamp TEXT NOT NULL
				);

				CREATE TABLE IF NOT EXISTS semantic_memory (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					category TEXT NOT NULL,
					text TEXT NOT NULL,
					summary TEXT NOT NULL,
					confidence REAL NOT NULL DEFAULT 0.5,
					tags TEXT NOT NULL DEFAULT '',
					timestamp TEXT NOT NULL
				);
				"""
			)

	def add_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
		payload = json.dumps(metadata or {}, ensure_ascii=False)
		with self._connect() as conn:
			cursor = conn.execute(
				"INSERT INTO turns(role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
				(role, content, utc_now_iso(), payload),
			)
			return int(cursor.lastrowid)

	def get_recent_turns(self, limit: int = MAX_SESSION_TURNS) -> List[ConversationTurn]:
		with self._connect() as conn:
			rows = conn.execute(
				"SELECT role, content, timestamp, metadata FROM turns ORDER BY id DESC LIMIT ?",
				(limit,),
			).fetchall()
		return [
			ConversationTurn(role=row["role"], content=row["content"], timestamp=row["timestamp"], metadata=safe_json_loads(row["metadata"], {}))
			for row in reversed(rows)
		]

	def set_preference(self, key: str, value: Any, confidence: float = 0.6) -> None:
		with self._connect() as conn:
			conn.execute(
				"""
				INSERT INTO preferences(key, value, confidence, updated_at)
				VALUES (?, ?, ?, ?)
				ON CONFLICT(key) DO UPDATE SET
					value=excluded.value,
					confidence=excluded.confidence,
					updated_at=excluded.updated_at
				""",
				(key, json.dumps(value, ensure_ascii=False), confidence, utc_now_iso()),
			)

	def get_preferences(self) -> Dict[str, Any]:
		with self._connect() as conn:
			rows = conn.execute("SELECT key, value, confidence FROM preferences").fetchall()
		return {
			row["key"]: {"value": safe_json_loads(row["value"], row["value"]), "confidence": float(row["confidence"])}
			for row in rows
		}

	def upsert_fact(self, category: str, key: str, value: Any, confidence: float = 0.5, source: str = "interaction") -> None:
		now = utc_now_iso()
		with self._connect() as conn:
			conn.execute(
				"""
				INSERT INTO facts(category, key, value, confidence, source, created_at, updated_at)
				VALUES (?, ?, ?, ?, ?, ?, ?)
				ON CONFLICT(category, key) DO UPDATE SET
					value=excluded.value,
					confidence=excluded.confidence,
					source=excluded.source,
					updated_at=excluded.updated_at
				""",
				(category, key, json.dumps(value, ensure_ascii=False), confidence, source, now, now),
			)

	def get_facts(self) -> Dict[str, Dict[str, Any]]:
		with self._connect() as conn:
			rows = conn.execute("SELECT category, key, value, confidence, source, updated_at FROM facts").fetchall()
		facts: Dict[str, Dict[str, Any]] = {}
		for row in rows:
			facts.setdefault(row["category"], {})[row["key"]] = {
				"value": safe_json_loads(row["value"], row["value"]),
				"confidence": float(row["confidence"]),
				"source": row["source"],
				"updated_at": row["updated_at"],
			}
		return facts

	def add_audit_log(self, action: str, detail: str, success: bool = True) -> None:
		with self._connect() as conn:
			conn.execute(
				"INSERT INTO audit_log(action, detail, success, timestamp) VALUES (?, ?, ?, ?)",
				(action, detail, 1 if success else 0, utc_now_iso()),
			)

	def add_mood_sample(self, mood: str, stress: float, focus: float, source: str = "text") -> None:
		with self._connect() as conn:
			conn.execute(
				"INSERT INTO mood_samples(mood, stress, focus, source, timestamp) VALUES (?, ?, ?, ?, ?)",
				(mood, stress, focus, source, utc_now_iso()),
			)

	def add_semantic_memory(self, category: str, text: str, summary: Optional[str] = None, confidence: float = 0.5, tags: Optional[List[str]] = None) -> int:
		payload_summary = summary or self._summarize_text(text)
		payload_tags = ",".join(tags or [])
		with self._connect() as conn:
			cursor = conn.execute(
				"INSERT INTO semantic_memory(category, text, summary, confidence, tags, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
				(category, text, payload_summary, confidence, payload_tags, utc_now_iso()),
			)
			return int(cursor.lastrowid)

	def search_semantic_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
		query_tokens = set(tokenise(query))
		if not query_tokens:
			return []
		with self._connect() as conn:
			rows = conn.execute(
				"SELECT category, text, summary, confidence, tags, timestamp FROM semantic_memory ORDER BY id DESC LIMIT 1000"
			).fetchall()
		results: List[Tuple[float, sqlite3.Row]] = []
		for row in rows:
			tokens = set(tokenise(f"{row['summary']} {row['text']} {row['tags']}"))
			overlap = len(tokens & query_tokens)
			if overlap:
				score = overlap / max(1, len(query_tokens))
				results.append((score, row))
		results.sort(key=lambda item: item[0], reverse=True)
		return [
			{
				"category": row["category"],
				"text": row["text"],
				"summary": row["summary"],
				"confidence": float(row["confidence"]),
				"tags": [tag for tag in row["tags"].split(",") if tag],
				"timestamp": row["timestamp"],
			}
			for _, row in results[:limit]
		]

	def list_reminders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
		with self._connect() as conn:
			if status:
				rows = conn.execute(
					"SELECT id, title, due_at, status, metadata, created_at, updated_at FROM reminders WHERE status=? ORDER BY id DESC",
					(status,),
				).fetchall()
			else:
				rows = conn.execute(
					"SELECT id, title, due_at, status, metadata, created_at, updated_at FROM reminders ORDER BY id DESC"
				).fetchall()
		return [
			{
				"id": int(row["id"]),
				"title": row["title"],
				"due_at": row["due_at"],
				"status": row["status"],
				"metadata": safe_json_loads(row["metadata"], {}),
				"created_at": row["created_at"],
				"updated_at": row["updated_at"],
			}
			for row in rows
		]

	def update_reminder_status(self, reminder_id: int, status: str) -> None:
		with self._connect() as conn:
			conn.execute("UPDATE reminders SET status=?, updated_at=? WHERE id=?", (status, utc_now_iso(), reminder_id))

	def log_feedback(self, turn_id: Optional[int], rating: float, note: str = "") -> None:
		with self._connect() as conn:
			conn.execute(
				"INSERT INTO learning_events(event_type, payload, timestamp) VALUES (?, ?, ?)",
				("feedback", json.dumps({"turn_id": turn_id, "rating": rating, "note": note}), utc_now_iso()),
			)

	def log_learning_event(self, event_type: str, payload: Dict[str, Any]) -> None:
		with self._connect() as conn:
			conn.execute(
				"INSERT INTO learning_events(event_type, payload, timestamp) VALUES (?, ?, ?)",
				(event_type, json.dumps(payload, ensure_ascii=False), utc_now_iso()),
			)

	def get_skill_stats(self) -> Dict[str, Any]:
		return {}

	def _summarize_text(self, text: str, max_words: int = 24) -> str:
		words = tokenise(text)
		if len(words) <= max_words:
			return text[:180]
		return " ".join(words[:max_words])


class UserModel:
	"""Learns your preferences, phrasing, and recurring behavior over time."""

	def __init__(self, memory: MemoryStore):
		self.memory = memory
		self.profile = {
			"preferred_length": "detailed",
			"formality": 0.5,
			"humor": 0.5,
			"verbosity": 0.5,
			"proactive": 0.5,
			"technical_depth": 0.5,
			"stress_sensitivity": 0.5,
			"response_style": "balanced",
		}
		self._load_from_memory()

	def _load_from_memory(self) -> None:
		prefs = self.memory.get_preferences()
		for key, item in prefs.items():
			if key in self.profile:
				self.profile[key] = item["value"]

	def observe_turn(self, user_text: str, signals: UserSignal, intent: str) -> None:
		text = normalize_text(user_text)
		if not text:
			return
		if len(tokenise(text)) >= 20:
			self.profile["preferred_length"] = "detailed"
		if signals.stress > 0.6:
			self.profile["stress_sensitivity"] = clamp(self.profile["stress_sensitivity"] * 0.9 + 0.1, 0.0, 1.0)
			self.profile["humor"] = clamp(self.profile["humor"] * 0.85, 0.0, 1.0)
		if signals.formality > 0.65:
			self.profile["formality"] = clamp(self.profile["formality"] * 0.85 + 0.15, 0.0, 1.0)
		if intent in {"research", "analysis", "planning"}:
			self.profile["technical_depth"] = clamp(self.profile["technical_depth"] * 0.9 + 0.1, 0.0, 1.0)
		self._persist_profile()

	def learn_from_feedback(self, rating: float, note: str = "") -> None:
		rating = clamp(rating, 0.0, 1.0)
		if rating >= 0.8:
			self.profile["humor"] = clamp(self.profile["humor"] + 0.03, 0.0, 1.0)
			self.profile["proactive"] = clamp(self.profile["proactive"] + 0.02, 0.0, 1.0)
		elif rating <= 0.4:
			self.profile["humor"] = clamp(self.profile["humor"] - 0.03, 0.0, 1.0)
			self.profile["verbosity"] = clamp(self.profile["verbosity"] - 0.02, 0.0, 1.0)
		self._persist_profile()
		self.memory.log_learning_event("feedback", {"rating": rating, "note": note})

	def _persist_profile(self) -> None:
		for key, value in self.profile.items():
			self.memory.set_preference(key, value, confidence=0.65)

	def profile_snapshot(self) -> Dict[str, Any]:
		return dict(self.profile)

	def infer_style(self, user_text: str, voice_energy: float = 0.5) -> UserSignal:
		text = normalize_text(user_text)
		tokens = tokenise(text)
		urgency_markers = {"now", "quick", "urgent", "asap", "fast", "immediately"}
		stress_markers = {"stressed", "anxious", "tired", "overwhelmed", "frustrated", "angry"}
		humor_markers = {"lol", "haha", "joke", "funny", "humor"}

		urgency = 1.0 if urgency_markers & set(tokens) else 0.1
		stress = 1.0 if stress_markers & set(tokens) else 0.2
		humor = 0.8 if humor_markers & set(tokens) else float(self.profile.get("humor", 0.5))
		formality = 0.65 if any(word in text for word in ["please", "kindly", "could you", "would you"]) else float(self.profile.get("formality", 0.5))
		focus = clamp(1.0 - urgency * 0.3, 0.0, 1.0)
		mood = "stressed" if stress > 0.6 else "focused" if focus > 0.6 else "neutral"
		text_style = "concise" if len(tokens) <= 8 else "detailed" if len(tokens) >= 20 else "balanced"
		return UserSignal(text_style=text_style, urgency=urgency, stress=stress, focus=focus, formality=formality, humor_preference=humor, task_intensity=voice_energy, mood=mood, source="text")


class IntentRouter:
	"""Routes input to the correct processing path."""

	TASK_PATTERNS = {
		"research": [r"\b(research|find|search|look up|lookup|compare|source|what is|who is|where is|tell me about|brief of|explain)\b"],
		"summary": [r"\b(summarize|summary|tl;dr|in short|short summary|brief summary|sum up)\b"],
		"system": [r"\b(open|close|minimise|minimize|launch|switch|resize|delete|move|rename|copy|paste)\b"],
		"memory": [r"\b(remember|forget|store|recall|what do you know)\b"],
		"analysis": [r"\b(analyse|analyze|explain|why|how does|difference|compare)\b"],
		"conversation": [r"\b(hi|hello|hey|how are you|what's up|chat)\b"],
		"planning": [r"\b(plan|roadmap|strategy|next steps|architecture)\b"],
	}
	COMMAND_STARTERS = {"open", "close", "launch", "delete", "move", "rename", "copy", "paste", "run", "start", "stop", "search", "find", "remember", "forget"}

	def route(self, text: str) -> Tuple[str, float, Dict[str, Any]]:
		normalized = normalize_text(text)
		if not normalized:
			return "empty", 0.0, {}
		if re.search(r"\b(model|which model|what model|backend model|llm)\b", normalized):
			return "system", 0.97, {"reason": "model identity request"}
		if re.search(r"\b(year|which year|current year|what year is this)\b", normalized):
			return "system", 0.97, {"reason": "local year request"}
		if re.search(r"\b(time|date|day|today'?s date|current time|current date|what time is it|what is the time|tell me time)\b", normalized):
			return "system", 0.96, {"reason": "local time/date request"}
		if re.search(r"\b(weather|forecast|temperature|rain|humidity|wind)\b", normalized):
			return "system", 0.95, {"reason": "local weather request"}
		if re.search(r"\b(gpu|graphics|vram|video memory|gpu usage)\b", normalized):
			return "system", 0.96, {"reason": "gpu status request"}
		if re.search(r"\b(ip address|public ip|local ip|my ip|internet ip|network ip|ip)\b", normalized):
			return "system", 0.96, {"reason": "ip address request"}
		if re.search(r"\b(traffic|troffic|trafic|congestion|jam|road traffic|route traffic|travel time|eta)\b", normalized):
			return "system", 0.95, {"reason": "city traffic request"}
		if re.search(r"\b(news|city news|local news|updates|local updates|headlines|current events)\b", normalized):
			return "system", 0.94, {"reason": "local city news request"}
		if re.search(r"\b(global news|worldwide news|world news|trending|tech news|crypto news|business news)\b", normalized):
			return "system", 0.94, {"reason": "global news request"}
		scores = Counter()
		for intent, patterns in self.TASK_PATTERNS.items():
			for pattern in patterns:
				if re.search(pattern, normalized):
					scores[intent] += 1
		if any(word in normalized for word in ["weather", "traffic", "troffic", "trafic", "news", "today", "current", "latest"]):
			scores["research"] += 1
		if any(word in normalized for word in ["open", "launch", "delete", "close", "move", "rename", "switch"]):
			scores["system"] += 1
		if any(word in normalized for word in ["gpu", "graphics", "vram", "ip", "network", "internet"]):
			scores["system"] += 2
		if any(word in normalized for word in ["youtube", "browser", "website", "site", "tab", "play", "pause", "mute", "fullscreen", "scroll"]):
			scores["system"] += 2
		if any(word in normalized for word in ["remember", "forget", "save this"]):
			scores["memory"] += 1
		if not scores:
			tokens = tokenise(normalized)
			if tokens and tokens[0] in self.COMMAND_STARTERS:
				return "command", 0.62, {"reason": "explicit command starter"}
			if normalized.endswith("?"):
				return "conversation", 0.58, {"reason": "question treated as conversation"}
			return "conversation", 0.45, {"reason": "default conversational path"}
		intent, score = scores.most_common(1)[0]
		confidence = clamp(0.4 + 0.15 * score, 0.0, 0.99)
		return intent, confidence, {"scores": dict(scores)}


class IntentClassifierLLM:
	"""Optional model-based intent classifier to strengthen routing quality."""

	VALID_INTENTS = {"system", "command", "research", "conversation", "summary", "memory", "analysis", "planning"}

	def __init__(self, model: str = DEFAULT_INTENT_MODEL):
		self.client = OpenAICompatibleLLMClient(model=model, timeout=10)
		self.model = model

	def classify(self, user_text: str, heuristic_intent: str, heuristic_confidence: float, heuristic_details: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
		# Preserve highly confident local/system routing decisions.
		if heuristic_intent == "system" and heuristic_confidence >= 0.94:
			return heuristic_intent, heuristic_confidence, {"source": "heuristic-locked", **(heuristic_details or {})}
		if heuristic_intent == "command" and heuristic_confidence >= 0.9:
			return heuristic_intent, heuristic_confidence, {"source": "heuristic-locked", **(heuristic_details or {})}
		if not self.client.enabled:
			return heuristic_intent, heuristic_confidence, {"source": "heuristic", **(heuristic_details or {})}
		prompt = {
			"role": "system",
			"content": (
				"Classify the user's intent into exactly one label: system, command, research, conversation, summary, memory, analysis, planning. "
				"Return strict JSON only: {\"intent\":\"...\",\"confidence\":0.0-1.0,\"reason\":\"...\"}. "
				"Interpret 'everyday normal chat' as conversation. "
				"Use summary only when user asks to summarize or shorten existing content."
			)
		}
		user_message = {
			"role": "user",
			"content": (
				f"User text: {user_text}\n"
				f"Heuristic intent: {heuristic_intent}\n"
				f"Heuristic confidence: {heuristic_confidence:.2f}\n"
				f"Heuristic details: {json.dumps(heuristic_details or {}, ensure_ascii=False)}"
			)
		}
		result = self.client.chat([prompt, user_message], temperature=0.0, top_p=1.0, max_tokens=120)
		if not result:
			return heuristic_intent, heuristic_confidence, {"source": "heuristic-fallback", **(heuristic_details or {})}
		parsed = self._parse_json(result)
		if not parsed:
			return heuristic_intent, heuristic_confidence, {"source": "heuristic-fallback", **(heuristic_details or {})}
		intent = str(parsed.get("intent", "")).strip().lower()
		if intent == "everyday":
			intent = "conversation"
		if intent not in self.VALID_INTENTS:
			return heuristic_intent, heuristic_confidence, {"source": "heuristic-fallback", **(heuristic_details or {})}
		confidence = float(parsed.get("confidence", heuristic_confidence))
		confidence = clamp(confidence, 0.0, 0.99)
		reason = str(parsed.get("reason", "")).strip()
		if confidence < 0.45:
			return heuristic_intent, heuristic_confidence, {"source": "heuristic-low-confidence", "model_reason": reason, **(heuristic_details or {})}
		return intent, max(confidence, heuristic_confidence * 0.85), {"source": "intent-model", "model": self.model, "model_reason": reason, "heuristic": heuristic_details or {}}

	def _parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
		text = raw.strip()
		try:
			return json.loads(text)
		except Exception:
			pass
		match = re.search(r"\{.*\}", text, flags=re.S)
		if not match:
			return None
		try:
			return json.loads(match.group(0))
		except Exception:
			return None


class PersonaEngine:
	"""Shapes responses to feel like Jarvis without shortening them."""

	OPENERS = ["Certainly.", "Understood.", "Right away.", "As you wish.", "Absolutely.", "All right."]

	def style_response(self, text: str, signal: UserSignal, profile: Dict[str, Any]) -> str:
		text = text.strip()
		if not text:
			return "I have nothing useful to report."
		lowered = text.lstrip().lower()
		if "\n" in text or lowered.startswith("{") or lowered.startswith("[") or lowered.startswith("research report for:"):
			return text
		if len(text) <= 180:
			return text
		formality = float(profile.get("formality", 0.5))
		humor = float(profile.get("humor", 0.5))
		if formality > 0.72:
			opener = self.OPENERS[0]
		elif signal.urgency > 0.55:
			opener = self.OPENERS[2]
		elif signal.stress > 0.55:
			opener = self.OPENERS[1]
		elif len(text) > 240:
			opener = self.OPENERS[4]
		else:
			opener = self.OPENERS[5]
		if humor > 0.7 and signal.stress < 0.5:
			opener = f"{opener} I’ll keep it lean."
		return f"{opener} {text}".strip()


class DialogueTuner:
	"""Fast-path dialogue handling for live, human-feeling exchanges."""

	QUICK_PATTERNS = [
		r"\b(hi|hello|hey|good morning|good evening|thanks|thank you|cool|nice|okay|ok)\b",
		r"\b(are you there|you there|what can you do|who are you|what are you)\b",
		r"\b(don't repeat|stop repeating|be natural|sound human|less robotic|not robotic|faster|speed up)\b",
	]

	def __init__(self) -> None:
		self._last_choice_by_bucket: Dict[str, str] = {}

	def _pick_variant(self, bucket: str, options: List[str]) -> str:
		if not options:
			return ""
		last = self._last_choice_by_bucket.get(bucket)
		pool = [item for item in options if item != last] if len(options) > 1 else options
		picked = random.choice(pool)
		self._last_choice_by_bucket[bucket] = picked
		return picked

	def should_bypass_llm(self, user_text: str, intent: str, talkative_level: str = "balanced") -> bool:
		if intent != "conversation":
			return False
		normalized = normalize_text(user_text)
		tokens = tokenise(normalized)
		if talkative_level == "peak" and len(tokens) >= 3:
			return False
		if not tokens:
			return True
		if len(tokens) <= 2:
			return True

		# Short but substantive questions should go through the model.
		question_markers = {
			"who", "what", "when", "where", "why", "how", "which", "explain", "details", "detail", "more", "about"
		}
		if normalized.endswith("?") and any(tok in question_markers for tok in tokens):
			return False
		if any(phrase in normalized for phrase in ["tell me", "explain", "more about", "do you know", "can you give", "in detail"]):
			return False

		if len(tokens) <= 6:
			return True
		if len(tokens) <= 12 and normalized.endswith("?"):
			return True
		if any(re.search(pattern, normalized) for pattern in self.QUICK_PATTERNS):
			return True
		return False

	def _localize(self, english: str, hinglish: str, language_mode: str, hindi: Optional[str] = None) -> str:
		if language_mode == "hinglish":
			return hinglish
		return english

	def quick_reply(self, user_text: str, signal: UserSignal, profile: Dict[str, Any], language_mode: str = "english") -> Optional[str]:
		normalized = normalize_text(user_text)
		tokens = set(tokenise(normalized))
		if not normalized:
			return self._localize(
				"I am right here with you. Share what is on your mind, and I will help you think it through step by step.",
				"Main yahin hoon. Jo bhi mind mein hai bolo, main usse step by step clear karne mein help karunga.",
				language_mode,
				"मैं यहीं हूँ। जो भी आपके मन में है बताइए, मैं उसे चरणबद्ध तरीके से सुलझाने में मदद करूँगा।",
			)
		if ({"hi", "hello", "hey"} & tokens) or "good morning" in normalized or "good evening" in normalized or "good afternoon" in normalized:
			eng = self._pick_variant("greeting_en", [
				"Hey, good to hear from you. We can chat casually, solve something quickly, or plan in detail. I can also pull local city news and updates where you are.",
				"Hello again. Want a quick action, a deep answer, or a clean step-by-step plan? I can fetch live updates from your current city too.",
				"Hi. I am fully online. Say one target and I will move fast on it, including local news and updates from your location.",
				"Hey there. Pick your mode: quick reply, research mode, or execution mode. For basics, you can also ask for local updates around you.",
			])
			hin = self._pick_variant("greeting_hi", [
				"Hey, achha laga sunke. Hum casual chat bhi kar sakte hain, koi problem solve bhi, ya detail plan bhi bana sakte hain. Main tumhari location ke local updates bhi de sakta hoon.",
				"Hello ji. Quick answer chahiye, deep research chahiye, ya direct execution? Chaaho to current city ka live update bhi laa doon.",
				"Hi, main fully online hoon. Ek clear target bolo, main turant handle karta hoon, including local city news aur updates.",
				"Namaste. Aaj speed mode, smart mode, ya deep mode mein kaam karein? Saath mein around-you updates bhi mil jayenge.",
			])
			return self._localize(eng, hin, language_mode)

		if any(phrase in normalized for phrase in ["how are you", "how are u", "what's up", "whats up", "kaise ho"]):
			eng = self._pick_variant("status_en", [
				"Running smooth and ready. What should we crush first? I can also give local updates from where you are.",
				"All systems stable. Give me one task and I will execute it cleanly, including city news around you.",
				"I am great and focused. Tell me your top priority right now, or ask for live updates near your location.",
			])
			hin = self._pick_variant("status_hi", [
				"Bilkul fit and ready. Sabse pehla target batao. Chaaho to tumhari jagah ke local updates bhi de doon.",
				"System full stable hai. Ek task bolo, clean execution deta hoon, city news ke saath.",
				"Main mast aur focused hoon. Ab tumhara top priority kya hai, ya location ke live updates chahiye?",
			])
			return self._localize(eng, hin, language_mode)
		if "thank you" in normalized or "thanks" in tokens:
			eng = self._pick_variant("thanks_en", [
				"Always. If you want, I can give a shorter version or a deeper version with context.",
				"Anytime. Want me to optimize that answer even further?",
				"You are welcome. Ready for the next one.",
			])
			hin = self._pick_variant("thanks_hi", [
				"Always. Chaaho to main short version ya context ke saath deep version de sakta hoon.",
				"Anytime. Chaaho to isko aur optimize kar doon?",
				"Welcome. Next command ke liye ready hoon.",
			])
			return self._localize(eng, hin, language_mode)
		if any(phrase in normalized for phrase in ["you there", "are you there", "what can you do", "who are you", "what are you"]):
			eng = self._pick_variant("capabilities_en", [
				"I am fully active and listening. I can talk naturally, research quickly, control system tasks, summarize long text, and help you decide between options.",
				"I can run browser commands, do quick research, draft content, summarize, and execute system actions fast.",
				"Think of me as your command bridge: voice in, action out, plus reasoning when needed.",
			])
			hin = self._pick_variant("capabilities_hi", [
				"Main fully active hoon aur sun raha hoon. Main natural baat kar sakta hoon, fast research kar sakta hoon, system tasks run kar sakta hoon, long text summarize kar sakta hoon, aur options mein decision help kar sakta hoon.",
				"Main browser commands chala sakta hoon, quick research kar sakta hoon, content draft kar sakta hoon, summary de sakta hoon aur system actions fast execute karta hoon.",
				"Mujhe voice command do, main action + smart reasoning ke saath output deta hoon.",
			])
			return self._localize(eng, hin, language_mode)

		if any(word in normalized for word in ["humor", "humour", "joke", "funny", "make me laugh", "hasao", "hansa"]):
			if any(phrase in normalized for phrase in ["dad joke", "papa joke"]):
				eng = self._pick_variant("dad_joke_en", [
					"Dad joke: I only know 25 letters of the alphabet. I do not know y.",
					"Dad joke: I used to be addicted to soap, but I am clean now.",
					"Dad joke: I would tell a construction joke, but I am still working on it.",
				])
				hin = self._pick_variant("dad_joke_hi", [
					"Dad joke: Alphabet ke 25 letters yaad hain, bas Y ka pata nahi.",
					"Dad joke: Pehle mujhe sabun ki aadat thi, ab bilkul clean hoon.",
					"Dad joke: Construction wali joke sunaata, par abhi us par kaam chal raha hai.",
				])
				return self._localize(eng, hin, language_mode)
			if any(phrase in normalized for phrase in ["one liner", "oneliner", "one-liner"]):
				eng = self._pick_variant("oneliner_en", [
					"One-liner: I am not lazy, I am on energy-saving mode.",
					"One-liner: My patience is like a browser tab count, dangerously high.",
					"One-liner: I debug like a detective with caffeine.",
				])
				hin = self._pick_variant("oneliner_hi", [
					"One-liner: Main lazy nahi, energy-saving mode mein hoon.",
					"One-liner: Meri patience browser tabs jaisi hai, bohot zyada.",
					"One-liner: Main debugging detective + caffeine combo se karta hoon.",
				])
				return self._localize(eng, hin, language_mode)
			if any(phrase in normalized for phrase in ["pun", "wordplay"]):
				eng = self._pick_variant("pun_en", [
					"Pun mode: I was going to tell a time-travel joke, but you did not like it.",
					"Pun mode: I am reading a book on anti-gravity, it is impossible to put down.",
					"Pun mode: I told my code to chill, now it has too many frozen threads.",
				])
				hin = self._pick_variant("pun_hi", [
					"Pun mode: Time-travel joke sunaata, par tumne kal hi mana kar diya.",
					"Pun mode: Anti-gravity ki book padh raha hoon, neeche rakhna impossible hai.",
					"Pun mode: Code ko chill bola, ab threads hi freeze ho gaye.",
				])
				return self._localize(eng, hin, language_mode)
			eng = self._pick_variant("joke_en", [
				"Quick joke: Why do programmers prefer dark mode? Because light attracts bugs.",
				"Quick joke: I told my laptop we needed a break, now it will not stop sleeping.",
				"Quick joke: Why was the function calm? It had no side effects.",
			])
			hin = self._pick_variant("joke_hi", [
				"Quick joke: Programmers dark mode kyun pasand karte hain? Kyunki light bugs ko attract karti hai.",
				"Quick joke: Laptop ko bola break chahiye, ab wo bas sleep hi karta rehta hai.",
				"Quick joke: Function calm kyun tha? Uske side effects nahi the.",
			])
			return self._localize(eng, hin, language_mode)

		if any(phrase in normalized for phrase in ["roast me", "insult me"]):
			eng = self._pick_variant("roast_en", [
				"Light roast: Your to-do list has more plot twists than a thriller series.",
				"Light roast: You open tabs like a collector, not a closer.",
				"Light roast: You say one quick task, then unlock a side quest chain.",
			])
			hin = self._pick_variant("roast_hi", [
				"Light roast: Tumhari to-do list mein thriller series se zyada plot twists hain.",
				"Light roast: Tum tabs close kam, collect zyada karte ho.",
				"Light roast: Ek quick task bolte ho, phir side quests ki series chalu ho jaati hai.",
			])
			return self._localize(eng, hin, language_mode)

		if any(phrase in normalized for phrase in ["motivate me", "i am tired", "feeling low", "demotivated"]):
			eng = self._pick_variant("motivate_en", [
				"You do not need perfect energy, only a clean next step. Start with a 5-minute sprint.",
				"Momentum beats mood. Pick one tiny action now, then we stack wins.",
				"Keep it simple: one task, one timer, zero overthinking.",
			])
			hin = self._pick_variant("motivate_hi", [
				"Perfect energy ki zarurat nahi, bas next clean step chahiye. 5-minute sprint se start karo.",
				"Mood se zyada momentum important hai. Ek chhota action lo, phir wins stack karte hain.",
				"Simple rakho: ek task, ek timer, zero overthinking.",
			])
			return self._localize(eng, hin, language_mode)

		if any(phrase in normalized for phrase in ["good night", "bye", "see you", "take care"]):
			eng = self._pick_variant("bye_en", [
				"Good night. Recharge well. I will be ready when you are back.",
				"Take care. Ping me anytime and we continue from here.",
				"Bye for now. Next session, we go faster and cleaner.",
			])
			hin = self._pick_variant("bye_hi", [
				"Good night. Achha rest lo. Tum wapas aaoge to main ready rahunga.",
				"Take care. Jab bhi ping karoge, yahin se continue karenge.",
				"Bye for now. Next session mein aur fast aur clean kaam karenge.",
			])
			return self._localize(eng, hin, language_mode)
		if any(phrase in normalized for phrase in ["don't repeat", "stop repeating", "be natural", "sound human", "less robotic", "not robotic", "faster", "speed up"]):
			eng = self._pick_variant("human_mode_en", [
				"Understood. I will keep the tone natural with lower repetition and tighter pacing.",
				"Done. From now on I will reply more human, less robotic, and faster.",
				"Got it. I will reduce repetitive lines and keep responses fresh.",
			])
			hin = self._pick_variant("human_mode_hi", [
				"Samajh gaya. Ab tone natural rahegi, repetition kam hogi, pacing tight rahegi.",
				"Done. Ab se replies zyada human, kam robotic, aur fast hongi.",
				"Got it. Repetitive lines kam karunga aur replies fresh rakhoonga.",
			])
			return self._localize(eng, hin, language_mode)
		if any(phrase in normalized for phrase in ["can you", "could you", "would you", "please"]):
			return None
		if signal.urgency > 0.65:
			return self._localize(
				"Right away. Give me the most critical target first, and I will execute in that order.",
				"Right away. Sabse critical target pehle bolo, main usi order mein execute karta hoon.",
				language_mode,
			)
		if signal.stress > 0.6:
			return self._localize(
				"Got it, no pressure. Keep it simple for now and I will guide the next steps one by one.",
				"Got it, tension mat lo. Abhi simple rakho, main next steps one by one guide karunga.",
				language_mode,
			)
		return None


class OpenAICompatibleLLMClient:
	"""Generic OpenAI-compatible chat client."""

	def __init__(self, base_url: str = DEFAULT_LLM_BASE_URL, api_key: str = DEFAULT_LLM_API_KEY, model: str = DEFAULT_LLM_MODEL, timeout: int = 30):
		self.base_url = base_url.rstrip("/")
		self.api_key = api_key
		self.model = model or DEFAULT_LLM_MODEL
		self.timeout = timeout
		self._sdk_client = None
		try:
			from openai import OpenAI  # type: ignore
			self._sdk_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
		except Exception:
			self._sdk_client = None

	@property
	def enabled(self) -> bool:
		return bool(self.base_url and self.api_key)

	def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, top_p: float = 1.0, max_tokens: int = 900) -> Optional[str]:
		if not self.enabled:
			return None
		if self._sdk_client is not None:
			try:
				completion = self._sdk_client.chat.completions.create(
					model=self.model,
					messages=messages,
					temperature=temperature,
					top_p=top_p,
					max_tokens=max_tokens,
					stream=False,
				)
				choice = completion.choices[0] if completion.choices else None
				message = getattr(choice, "message", None)
				content = getattr(message, "content", None) if message is not None else None
				return str(content).strip() if content else None
			except Exception:
				pass
		payload = json.dumps({"model": self.model, "messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}, ensure_ascii=False).encode("utf-8")
		request = urllib.request.Request(self.base_url + "/chat/completions", data=payload, method="POST", headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
		try:
			with urllib.request.urlopen(request, timeout=self.timeout) as response:
				data = json.loads(response.read().decode("utf-8"))
				choices = data.get("choices", [])
				if not choices:
					return None
				message = choices[0].get("message", {})
				return str(message.get("content", "")).strip() or None
		except Exception:
			return None


class RuleBasedLLMClient:
	"""Local fallback that synthesizes useful Jarvis replies without an API."""

	def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, top_p: float = 1.0, max_tokens: int = 900) -> str:
		user_text = messages[-1].get("content", "") if messages else ""
		normalized = normalize_text(user_text)
		insights: List[str] = []
		if any(word in normalized for word in ["hello", "hi", "hey", "good evening", "good morning"]):
			return "Systems are online. I am ready to assist, research, execute a command, and fetch local news or updates from your current location."
		if any(word in normalized for word in ["where are you", "are you there", "you there"]):
			return "I am active in your local session and ready for the next instruction."
		if any(word in normalized for word in ["thank you", "thanks", "very nice", "great", "awesome"]):
			return "Pleasure to help. If you want, I can continue with the next step."
		if any(word in user_text.lower() for word in ["plan", "roadmap", "steps"]):
			insights.append("I can break this into phases, dependencies, risks, and execution order.")
		if any(word in user_text.lower() for word in ["research", "find", "search", "what is", "who is", "where is", "tell me about", "brief of", "explain"]):
			insights.append("I should retrieve evidence first and then present the result in full detail.")
		if any(word in user_text.lower() for word in ["open", "close", "delete", "move"]):
			insights.append("This sounds like a tool-backed action and may require confirmation.")
		if not insights:
			insights.append("I am listening. Ask for research, system control, planning, memory, or conversation.")
		return " ".join(insights)


class LLMOrchestrator:
	"""Chooses model(s) by task and falls back safely when a model is unavailable."""

	def __init__(self, config: JarvisConfig):
		self.config = config
		self.remote = OpenAICompatibleLLMClient()
		# Backward-compatible alias used elsewhere.
		self.client = self.remote
		self.router_remote = OpenAICompatibleLLMClient(model=DEFAULT_INTENT_MODEL, timeout=6)
		self.fast_remote = OpenAICompatibleLLMClient(model=DEFAULT_FAST_MODEL, timeout=12)
		self.deep_remote = OpenAICompatibleLLMClient(model=DEFAULT_DEEP_MODEL, timeout=30)
		self.planning_remote = OpenAICompatibleLLMClient(model=DEFAULT_PLANNING_MODEL, timeout=25)
		self.coder_remote = OpenAICompatibleLLMClient(model=DEFAULT_CODER_MODEL, timeout=25)
		self._model_clients: Dict[str, OpenAICompatibleLLMClient] = {}
		self._route_cache: Dict[Tuple[str, str], str] = {}
		self._route_hits: Counter = Counter()
		self._recent_routes: Deque[Tuple[str, str]] = deque(maxlen=120)
		self.fallback = RuleBasedLLMClient()

	def _heuristic_route_task(self, original_task: str, user_text: str) -> Optional[str]:
		"""Ultra-fast local router to avoid LLM routing when intent is obvious."""
		task_key = normalize_text(original_task)
		text = normalize_text(user_text)
		if not text:
			return task_key

		if re.search(r"\b(code|python|javascript|bug|debug|refactor|function|class|api|sql|regex)\b", text):
			return "coding"
		if re.search(r"\b(plan|roadmap|milestone|timeline|strategy|execution plan|phases)\b", text):
			return "planning"
		if re.search(r"\b(tl;dr|summarize|summary|in short|brief summary|sum up)\b", text):
			return "summary"
		if re.search(r"\b(with sources|cite sources|deep research|detailed research|evidence)\b", text):
			return "research_deep"
		if re.search(r"\b(research|find|search|look up|what is|who is|where is|explain|analyze|analysis|compare)\b", text):
			return "research"
		if len(tokenise(text)) <= 5 and re.search(r"\b(hi|hello|hey|thanks|thank you|ok|okay|cool|nice|bye)\b", text):
			return "conversation_fast"
		return None

	def _route_task_fast(self, original_task: str, user_text: str, reasoning_mode: str) -> str:
		"""Use a small router model to choose the best task lane quickly."""
		task_key = normalize_text(original_task)
		text_key = normalize_text(user_text)[:220]
		cache_key = (task_key, text_key)
		cached = self._route_cache.get(cache_key)
		if cached:
			return cached

		heuristic = self._heuristic_route_task(task_key, text_key)
		if heuristic:
			if len(self._route_cache) >= 256:
				self._route_cache.clear()
			self._route_cache[cache_key] = heuristic
			return heuristic

		if not self.router_remote.enabled:
			return task_key

		if task_key in {"intent", "classification", "router", "conversation_fast"}:
			return task_key
		if len(tokenise(text_key)) <= 3:
			return task_key

		system_prompt = {
			"role": "system",
			"content": (
				"You are a fast task router. Pick one task label only from: "
				"conversation_fast, conversation, planning, coding, research, research_deep, analysis, summary. "
				"Return strict JSON only: {\"task\":\"...\"}."
			)
		}
		user_prompt = {
			"role": "user",
			"content": (
				f"Original task: {task_key}\n"
				f"Reasoning mode: {reasoning_mode}\n"
				f"User text: {user_text[:500]}"
			)
		}

		routed = self.router_remote.chat([system_prompt, user_prompt], temperature=0.0, top_p=1.0, max_tokens=36)
		if not routed:
			return task_key
		parsed = safe_json_loads(routed, {}) if isinstance(routed, str) else {}
		picked = normalize_text(str(parsed.get("task", ""))) if isinstance(parsed, dict) else ""
		valid = {"conversation_fast", "conversation", "planning", "coding", "research", "research_deep", "analysis", "summary"}
		final_task = picked if picked in valid else task_key

		# Small bounded cache for low-latency repeated commands.
		if len(self._route_cache) >= 256:
			self._route_cache.clear()
		self._route_cache[cache_key] = final_task
		return final_task

	def _task_generation_profile(self, task: str, reasoning_mode: str, temperature: float, max_tokens: int) -> Tuple[float, int]:
		"""Apply lightweight per-task generation tuning for better quality/speed balance."""
		task_key = normalize_text(task)
		profiles: Dict[str, Tuple[float, int]] = {
			"conversation_fast": (0.45, 220),
			"conversation": (0.5, 520),
			"planning": (0.25, 900),
			"coding": (0.2, 900),
			"summary": (0.2, 420),
			"research": (0.22, 820),
			"research_deep": (0.15, 1200),
			"analysis": (0.2, 900),
		}
		profile_temp, profile_tokens = profiles.get(task_key, (0.3, 700))

		if reasoning_mode == "fast":
			profile_tokens = min(profile_tokens, 380)
			profile_temp = max(profile_temp, 0.35)
		elif reasoning_mode == "deep":
			profile_tokens = max(profile_tokens, 900)

		effective_temp = profile_temp if abs(temperature - 0.2) < 1e-9 else temperature
		effective_tokens = profile_tokens if max_tokens == 900 else max_tokens
		return effective_temp, effective_tokens

	def _get_model_client(self, model_name: str, timeout: int = 24) -> OpenAICompatibleLLMClient:
		if model_name not in self._model_clients:
			self._model_clients[model_name] = OpenAICompatibleLLMClient(model=model_name, timeout=timeout)
		return self._model_clients[model_name]

	def select_model(self, task: str, reasoning_mode: str = "balanced") -> str:
		return self.candidate_models_for_task(task, reasoning_mode)[0]

	def candidate_models_for_task(self, task: str, reasoning_mode: str = "balanced") -> List[str]:
		task_key = normalize_text(task)

		if task_key in {"intent", "classification", "router"}:
			ordered = [DEFAULT_INTENT_MODEL, DEFAULT_FAST_MODEL, DEFAULT_PLANNING_MODEL]
		elif task_key in {"conversation_fast", "smalltalk", "quick"}:
			ordered = [DEFAULT_FAST_MODEL, DEFAULT_LLM_MODEL, DEFAULT_PLANNING_MODEL]
		elif task_key in {"planning", "strategy", "roadmap"}:
			ordered = [DEFAULT_PLANNING_MODEL, DEFAULT_REASONING_MODEL, DEFAULT_LLM_MODEL, DEFAULT_FAST_MODEL]
		elif task_key in {"coding", "code", "debug", "refactor"}:
			ordered = [DEFAULT_CODER_MODEL, DEFAULT_LLM_MODEL, DEFAULT_REASONING_MODEL, DEFAULT_FAST_MODEL]
		elif task_key in {"research_deep"}:
			ordered = [DEFAULT_LONG_CONTEXT_MODEL, DEFAULT_REASONING_MODEL, DEFAULT_DEEP_MODEL, DEFAULT_LLM_MODEL, DEFAULT_FAST_MODEL]
		elif task_key in {"research", "analysis", "summary"}:
			ordered = [DEFAULT_REASONING_MODEL, DEFAULT_DEEP_MODEL, DEFAULT_LONG_CONTEXT_MODEL, DEFAULT_LLM_MODEL, DEFAULT_FAST_MODEL]
		elif reasoning_mode == "fast":
			ordered = [DEFAULT_FAST_MODEL, DEFAULT_LLM_MODEL, DEFAULT_PLANNING_MODEL]
		elif reasoning_mode == "deep":
			ordered = [DEFAULT_LONG_CONTEXT_MODEL, DEFAULT_REASONING_MODEL, DEFAULT_DEEP_MODEL, DEFAULT_LLM_MODEL, DEFAULT_FAST_MODEL]
		else:
			ordered = [DEFAULT_LLM_MODEL, DEFAULT_PLANNING_MODEL, DEFAULT_REASONING_MODEL, DEFAULT_FAST_MODEL]

		# Preserve order while removing duplicates.
		unique_models: List[str] = []
		for model_name in ordered:
			if model_name and model_name not in unique_models:
				unique_models.append(model_name)
		return unique_models

	def complete_for_task(self, task: str, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 900, reasoning_mode: str = "balanced") -> str:
		routed_task = task
		if messages:
			last_user = ""
			for msg in reversed(messages):
				if msg.get("role") == "user":
					last_user = msg.get("content", "")
					break
			if last_user:
				routed_task = self._route_task_fast(task, last_user, reasoning_mode)
		effective_temp, effective_tokens = self._task_generation_profile(routed_task, reasoning_mode, temperature, max_tokens)
		self._route_hits[routed_task] += 1
		self._recent_routes.append((normalize_text(task), normalize_text(routed_task)))

		candidate_models = self.candidate_models_for_task(routed_task, reasoning_mode)

		for model_name in candidate_models:
			client = self._get_model_client(model_name)
			if not client.enabled:
				continue
			result = client.chat(messages, temperature=effective_temp, top_p=0.8, max_tokens=effective_tokens)
			if result:
				return result
		return self.fallback.chat(messages, temperature=effective_temp, top_p=0.8, max_tokens=effective_tokens)

	def pipeline_overview(self) -> Dict[str, Any]:
		top_routes = dict(self._route_hits.most_common(6))
		return {
			"default": DEFAULT_LLM_MODEL,
			"fast": DEFAULT_FAST_MODEL,
			"deep": DEFAULT_DEEP_MODEL,
			"planning": DEFAULT_PLANNING_MODEL,
			"coding": DEFAULT_CODER_MODEL,
			"reasoning": DEFAULT_REASONING_MODEL,
			"long_context": DEFAULT_LONG_CONTEXT_MODEL,
			"router": DEFAULT_INTENT_MODEL,
			"route_hits": top_routes,
		}

	def complete(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 900) -> str:
		return self.complete_for_task("conversation", messages, temperature=temperature, max_tokens=max_tokens, reasoning_mode="balanced")

	def complete_fast(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 280) -> str:
		return self.complete_for_task("conversation_fast", messages, temperature=temperature, max_tokens=max_tokens, reasoning_mode="fast")


class ResearchService:
	"""Retrieval-first research helper."""

	def __init__(self, llm: LLMOrchestrator):
		self.llm = llm
		self._generic_reply_markers = ["systems are online", "ready to assist", "good to have you here"]
		self._answer_cache: Dict[Tuple[str, int], Tuple[float, Dict[str, Any]]] = {}
		self._search_cache: Dict[Tuple[str, int], Tuple[float, List[Dict[str, Any]]]] = {}
		self._wiki_cache: Dict[str, Tuple[float, str]] = {}
		self._page_cache: Dict[str, Tuple[float, str]] = {}
		self._cache_ttl_seconds = 3600.0

	def _cache_get(self, cache: Dict[Any, Tuple[float, Any]], key: Any) -> Any:
		entry = cache.get(key)
		if not entry:
			return None
		timestamp, value = entry
		if time.time() - timestamp > self._cache_ttl_seconds:
			cache.pop(key, None)
			return None
		return value

	def _cache_set(self, cache: Dict[Any, Tuple[float, Any]], key: Any, value: Any) -> None:
		cache[key] = (time.time(), value)

	def _normalize_result_url(self, href: str) -> str:
		url = html_unescape(href or "").strip()
		if not url:
			return ""
		if url.startswith("//"):
			url = "https:" + url
		parsed = urllib.parse.urlparse(url)
		if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
			params = urllib.parse.parse_qs(parsed.query)
			target = params.get("uddg", [""])[0]
			if target:
				return urllib.parse.unquote(target)
		return url

	def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
		cache_key = (normalize_text(query), max_results)
		cached = self._cache_get(self._search_cache, cache_key)
		if cached is not None:
			return cached
		encoded = urllib.parse.quote_plus(query)
		url = f"https://duckduckgo.com/html/?q={encoded}"
		try:
			request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
			with urllib.request.urlopen(request, timeout=20) as response:
				html_text = response.read().decode("utf-8", errors="ignore")
		except Exception:
			return []
		results: List[Dict[str, Any]] = []
		for match in re.finditer(r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', html_text, flags=re.S | re.I):
			href = self._normalize_result_url(match.group("href"))
			if not href:
				continue
			title = re.sub(r"<[^>]+>", "", match.group("title"))
			results.append({"title": html_unescape(title), "url": href})
			if len(results) >= max_results:
				break
		self._cache_set(self._search_cache, cache_key, results)
		return results

	def fetch_page_text(self, url: str) -> str:
		cached = self._cache_get(self._page_cache, url)
		if cached is not None:
			return cached
		try:
			request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
			with urllib.request.urlopen(request, timeout=20) as response:
				html_text = response.read().decode("utf-8", errors="ignore")
		except Exception:
			return ""
		text = re.sub(r"<script.*?</script>", " ", html_text, flags=re.S | re.I)
		text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
		text = re.sub(r"<[^>]+>", " ", text)
		result = re.sub(r"\s+", " ", text).strip()
		self._cache_set(self._page_cache, url, result)
		return result

	def wikipedia_lookup(self, query: str) -> str:
		cache_key = normalize_text(query)
		cached = self._cache_get(self._wiki_cache, cache_key)
		if cached is not None:
			return cached
		api = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(query.replace(" ", "_"))
		try:
			request = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
			with urllib.request.urlopen(request, timeout=20) as response:
				data = json.loads(response.read().decode("utf-8"))
				result = data.get("extract", "") or data.get("description", "") or ""
				self._cache_set(self._wiki_cache, cache_key, result)
				return result
		except Exception:
			return ""

	def summarize(self, query: str, sources: List[Dict[str, Any]]) -> str:
		chunks: List[str] = []
		for item in sources[:3]:
			text = self.fetch_page_text(item["url"])
			if text:
				chunks.append(text[:2500])
		wiki = self.wikipedia_lookup(query)
		messages = [
			{"role": "system", "content": "Answer the query in full detail using the provided source material. Do not summarize into a short brief. Preserve important facts, context, and caveats. If evidence is weak or conflicting, say so clearly."},
			{"role": "user", "content": f"Query: {query}\n\nSource excerpts:\n" + "\n\n".join(chunks) + f"\n\nWikipedia context: {wiki}"},
		]
		llm_text = (self.llm.complete_for_task("research_deep", messages, temperature=0.15, max_tokens=900, reasoning_mode="deep") or "").strip()
		normalized = normalize_text(llm_text)
		if not llm_text or any(marker in normalized for marker in self._generic_reply_markers):
			return self._heuristic_report(query, sources, wiki, chunks)
		return llm_text

	def fast_answer(self, query: str) -> Dict[str, Any]:
		cache_key = ("fast", normalize_text(query))
		cached = self._cache_get(self._answer_cache, cache_key)
		if cached is not None:
			return cached
		messages = [
			{"role": "system", "content": "Give a direct factual answer quickly. Keep it concise but useful. If uncertain, say what is uncertain. Do not browse or mention sources unless asked."},
			{"role": "user", "content": query},
		]
		answer = self.llm.complete_for_task("conversation_fast", messages, temperature=0.2, max_tokens=260, reasoning_mode="fast")
		result = {"answer": answer.strip() if answer else "I could not answer that quickly right now.", "sources": []}
		self._cache_set(self._answer_cache, cache_key, result)
		return result

	def _heuristic_report(self, query: str, sources: List[Dict[str, Any]], wiki: str, chunks: List[str]) -> str:
		lines = [f"Research report for: {query}", ""]
		if wiki:
			lines.append("Overview:")
			lines.append(wiki[:520].strip())
			lines.append("")
		if chunks:
			lines.append("Source excerpts:")
			for idx, text in enumerate(chunks[:3], start=1):
				snippet = re.sub(r"\s+", " ", text[:520]).strip()
				if snippet:
					lines.append(f"{idx}. {snippet}")
			lines.append("")
		if sources:
			lines.append("Sources:")
			for idx, item in enumerate(sources[:5], start=1):
				lines.append(f"{idx}. {item.get('title', 'Untitled')}")
				if item.get("url"):
					lines.append(f"   {item['url']}")
			lines.append("")
		if chunks:
			tokens: List[str] = []
			for text in chunks[:3]:
				tokens.extend(tokenise(text[:1200]))
			common = [word for word, _ in Counter(tokens).most_common(12) if len(word) > 4]
			if common:
				lines.append("Observed themes: " + ", ".join(common[:8]))
		lines.append("Note: This answer uses live retrieved sources and a heuristic fallback when the model output is too generic.")
		return "\n".join(lines).strip()

	def answer(self, query: str, max_results: int = 5) -> Dict[str, Any]:
		cache_key = (normalize_text(query), max_results)
		cached = self._cache_get(self._answer_cache, cache_key)
		if cached is not None:
			return cached
		sources = self.search_web(query, max_results=max_results)
		if not sources:
			wiki = self.wikipedia_lookup(query)
			if wiki:
				result = {"answer": wiki, "sources": [{"title": "Wikipedia", "url": "https://en.wikipedia.org"}]}
				self._cache_set(self._answer_cache, cache_key, result)
				return result
			result = {"answer": "I could not retrieve live sources right now.", "sources": []}
			self._cache_set(self._answer_cache, cache_key, result)
			return result
		result = {"answer": self.summarize(query, sources), "sources": sources}
		self._cache_set(self._answer_cache, cache_key, result)
		return result

	def daily_briefing(self) -> str:
		headlines = self.search_web("top headlines today", max_results=3)
		parts = ["Daily briefing:"]
		for item in headlines:
			parts.append(f"- {item['title']}")
		return "\n".join(parts)


class SystemController:
	"""System tasks on the local PC."""

	def __init__(self) -> None:
		self._last_youtube_query: Optional[str] = None

	WINDOWS_APP_ALIASES = {
		"chrome": [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", "chrome"],
		"google chrome": [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", "chrome"],
		"edge": [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe", "msedge"],
		"notepad": ["notepad.exe", "notepad"],
		"calculator": ["calc.exe", "calc"],
		"cmd": ["cmd.exe", "cmd"],
	}

	WEB_SHORTCUTS = {
		"google": "https://www.google.com",
		"youtube": "https://www.youtube.com",
		"gmail": "https://mail.google.com",
		"instagram": "https://www.instagram.com",
		"insta": "https://www.instagram.com",
		"facebook": "https://www.facebook.com",
		"x": "https://x.com",
		"twitter": "https://x.com",
		"linkedin": "https://www.linkedin.com",
		"reddit": "https://www.reddit.com",
		"whatsapp": "https://web.whatsapp.com",
		"telegram": "https://web.telegram.org",
	}

	def open_app(self, target: str) -> Dict[str, Any]:
		target = target.strip().strip('"').strip("'")
		if not target:
			return {"ok": False, "error": "empty target"}
		if re.search(r"[&|><^]", target):
			return {"ok": False, "error": "unsafe characters in target"}
		target_lower = target.lower()
		if any(name in target_lower for name in ["instagram", "twitter", "x", "facebook", "linkedin", "youtube", "gmail", "whatsapp", "telegram", "reddit"]):
			return self.open_url(self.WEB_SHORTCUTS.get(target_lower, next((url for name, url in self.WEB_SHORTCUTS.items() if name in target_lower), f"https://{target_lower}")))
		try:
			if platform.system().lower() == "windows":
				resolved = self._resolve_windows_target(target_lower, target)
				if resolved and Path(resolved).exists():
					os.startfile(resolved)
				else:
					subprocess.Popen(["cmd", "/c", "start", "", resolved or target], shell=False)
			else:
				executable = shutil.which(target) or target
				subprocess.Popen([executable])
			return {"ok": True, "target": target}
		except Exception as exc:
			return {"ok": False, "error": str(exc)}

	def _resolve_windows_target(self, target_lower: str, original_target: str) -> str:
		for candidate in self.WINDOWS_APP_ALIASES.get(target_lower, [original_target]):
			if ":\\" in candidate and Path(candidate).exists():
				return candidate
			if shutil.which(candidate):
				return candidate
		return original_target

	def open_url(self, url: str) -> Dict[str, Any]:
		webbrowser.open(url)
		return {"ok": True, "url": url}

	def _open_first_youtube_result(self, query: str) -> Dict[str, Any]:
		"""Open first YouTube search result with autoplay enabled."""
		search_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
		try:
			headers = {"User-Agent": "Mozilla/5.0"}
			resp = requests.get(search_url, headers=headers, timeout=8)
			resp.raise_for_status()
			video_ids = re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', resp.text)
			if not video_ids:
				watch_paths = re.findall(r"/watch\\?v=([A-Za-z0-9_-]{11})", resp.text)
				video_ids = watch_paths
			if video_ids:
				video_id = next((vid for vid in video_ids if len(vid) == 11), video_ids[0])
				watch_url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
				return {"ok": True, "action": "open_url", "url": watch_url, "result": self.open_url(watch_url)}
		except Exception:
			pass
		# Fallback: open search page and try selecting first result via keyboard.
		open_result = self.open_url(search_url)
		try:
			import keyboard  # type: ignore
			time.sleep(2.0)
			for _ in range(6):
				keyboard.send("tab")
				time.sleep(0.06)
			keyboard.send("enter")
			return {
				"ok": True,
				"action": "open_search_then_play_first",
				"url": search_url,
				"result": open_result,
			}
		except Exception:
			return {"ok": True, "action": "open_url", "url": search_url, "result": open_result}

	def _press_key(self, key_combo: str) -> Dict[str, Any]:
		try:
			import keyboard  # type: ignore
			keyboard.send(key_combo)
			return {"ok": True, "action": "key", "key": key_combo}
		except Exception as exc:
			return {"ok": False, "error": f"keyboard module unavailable or blocked: {exc}"}

	def _press_key_repeated(self, key_combo: str, count: int) -> Dict[str, Any]:
		try:
			import keyboard  # type: ignore
			for _ in range(max(1, count)):
				keyboard.send(key_combo)
			return {"ok": True, "action": "key_repeat", "key": key_combo, "count": count}
		except Exception as exc:
			return {"ok": False, "error": f"keyboard module unavailable or blocked: {exc}"}

	def _play_first_result_from_current_youtube_page(self) -> Dict[str, Any]:
		"""Try selecting first visible YouTube result from current tab using keyboard navigation."""
		try:
			import keyboard  # type: ignore
			for _ in range(6):
				keyboard.send("tab")
				time.sleep(0.06)
			keyboard.send("enter")
			return {"ok": True, "action": "key", "key": "tabx6+enter"}
		except Exception as exc:
			return {"ok": False, "error": f"keyboard module unavailable or blocked: {exc}"}

	def _scroll_page(self, direction: str, count: int = 1) -> Dict[str, Any]:
		"""Scroll the current page by repeated page up/down presses."""
		key = "pagedown" if direction == "down" else "pageup"
		return self._press_key_repeated(key, count)

	def _set_youtube_volume_fast(self, level: int) -> Dict[str, Any]:
		"""Set YouTube volume approximately using fast key sequences.

		YouTube volume changes in 5 percent steps with arrow keys, so this uses coarse presets.
		"""
		level = max(0, min(100, int(level)))
		if level == 0:
			return self._press_key("m")
		steps_map = {
			20: 4,
			40: 8,
			60: 12,
			80: 16,
			100: 20,
		}
		steps = steps_map.get(level)
		if steps is None:
			# Round to nearest 20% block for predictable behavior.
			rounded = min(100, max(0, int(round(level / 20.0) * 20)))
			if rounded == 0:
				return self._press_key("m")
			steps = steps_map.get(rounded, 8)
		# Try to avoid huge jumps by using mute + up presses when setting low levels.
		if level <= 20:
			result = self._press_key("m")
			if not result.get("ok"):
				return result
			return self._press_key_repeated("up", steps)
		return self._press_key_repeated("up", steps)

	def get_weather_report(self, location: Optional[str] = None) -> Dict[str, Any]:
		"""Fetch current weather from wttr.in without needing an API key."""
		try:
			place = location.strip() if location else ""
			if place:
				alias_map = {
					"baroda": "vadodara",
					"bombay": "mumbai",
					"calcutta": "kolkata",
					"madras": "chennai",
					"bangalore": "bengaluru",
				}
				normalized_place = normalize_text(place)
				place = alias_map.get(normalized_place, place)
			url = f"https://wttr.in/{urllib.parse.quote(place)}?format=j1" if place else "https://wttr.in/?format=j1"
			resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
			resp.raise_for_status()
			# Fetch live gold and silver prices in INR
			try:
				# Use metals.live free API (no key required)
				metals_resp = requests.get(
					"https://api.metals.live/v1/spot/gold,silver",
					headers={"User-Agent": "Mozilla/5.0"},
					timeout=6,
				)
				metals_resp.raise_for_status()
				metals_data = metals_resp.json()
				# Response is a list of {symbol, price} objects in USD per troy oz
				metals_map = {item["symbol"]: item["price"] for item in metals_data if "symbol" in item and "price" in item}
				# Fetch USD→INR rate
				fx_resp = requests.get(
					"https://api.exchangerate-api.com/v4/latest/USD",
					headers={"User-Agent": "Mozilla/5.0"},
					timeout=5,
				)
				fx_resp.raise_for_status()
				usd_to_inr = float(fx_resp.json()["rates"]["INR"])
				gold_usd = float(metals_map.get("XAU", metals_map.get("gold", 0)))
				silver_usd = float(metals_map.get("XAG", metals_map.get("silver", 0)))
				if gold_usd > 0:
					gold_price = f"{gold_usd * usd_to_inr:,.0f}"
				else:
					gold_price = "unavailable"
				if silver_usd > 0:
					silver_price = f"{silver_usd * usd_to_inr:,.0f}"
				else:
					silver_price = "unavailable"
			except Exception:
				gold_price = "unavailable"
				silver_price = "unavailable"
			data = resp.json()
			current = (data.get("current_condition") or [{}])[0]
			daily = data.get("weather") or []
			today = daily[0] if len(daily) >= 1 else {}
			tomorrow = daily[1] if len(daily) >= 2 else {}
			area = (((data.get("nearest_area") or [{}])[0]).get("areaName") or [{}])[0].get("value", place or "your location")
			region = (((data.get("nearest_area") or [{}])[0]).get("region") or [{}])[0].get("value", "")
			temp_c = current.get("temp_C", "?")
			feels_like = current.get("FeelsLikeC", "?")
			humidity = current.get("humidity", "?")
			wind_kmph = current.get("windspeedKmph", "?")
			desc = ((current.get("weatherDesc") or [{}])[0]).get("value", "Unknown")
			today_max = today.get("maxtempC", "?")
			today_min = today.get("mintempC", "?")
			today_hourly = today.get("hourly") or []
			if today_hourly:
				rain_chances = [int(h.get("chanceofrain", 0)) for h in today_hourly]
				today_rain = max(rain_chances)
			else:
				today_rain = "N/A"
			tomorrow_hourly = tomorrow.get("hourly") or []
			tomorrow_desc = "Unknown"
			if tomorrow_hourly:
				tomorrow_desc = ((tomorrow_hourly[0].get("weatherDesc") or [{}])[0]).get("value", "Unknown")
			parts = [
				f"Weather for {area}" + (f", {region}" if region else "") + ":",
				f"Condition: {desc}",
				f"Temperature: {temp_c}°C",
				f"Feels like: {feels_like}°C",
				f"Humidity: {humidity}%",
				f"Wind: {wind_kmph} km/h",
				f"Forecast today: High {today_max}°C, Low {today_min}°C, Rain chance {today_rain}%",
				f"Forecast tomorrow: {tomorrow_desc}",
				f"Gold (24K): ₹{gold_price} per oz",
				f"Silver: ₹{silver_price} per oz",
			]
			return {"ok": True, "action": "weather", "location": area, "report": "\n".join(parts)}
		except Exception as exc:
			return {"ok": False, "error": f"weather unavailable: {exc}"}

	def _geocode_city(self, city: str) -> Optional[Dict[str, Any]]:
		try:
			url = "https://nominatim.openstreetmap.org/search"
			params = {
				"q": city,
				"format": "jsonv2",
				"limit": 1,
			}
			resp = requests.get(url, params=params, headers={"User-Agent": "Jarvis/1.0 (+local assistant)"}, timeout=8)
			resp.raise_for_status()
			rows = resp.json() or []
			if not rows:
				return None
			row = rows[0]
			lat = float(row.get("lat"))
			lon = float(row.get("lon"))
			name = str(row.get("display_name", city)).split(",")[0].strip() or city
			return {"lat": lat, "lon": lon, "name": name}
		except Exception:
			return None

	def get_city_traffic_report(self, city: Optional[str] = None) -> Dict[str, Any]:
		"""Open city traffic in Google Maps (no API key required)."""
		try:
			place = (city or "").strip() or "my location"
			geo = self._geocode_city(place) if place != "my location" else None
			if place != "my location" and not geo:
				maps_url = "https://www.google.com/maps/search/" + urllib.parse.quote_plus(f"traffic in {place}")
				self.open_url(maps_url)
				return {
					"ok": False,
					"action": "traffic",
					"city": place,
					"error": f"could not resolve city '{place}'",
					"fallback_url": maps_url,
					"report": f"I could not resolve {place}. I opened live traffic search in Google Maps.",
				}
			city_name = (geo or {}).get("name", place)
			maps_url = "https://www.google.com/maps/search/" + urllib.parse.quote_plus(f"traffic in {city_name}")
			self.open_url(maps_url)
			report_lines = [
				f"Live traffic opened for {city_name} in Google Maps.",
				"Google Maps will show current congestion colors and alternate roads.",
				f"Map: {maps_url}",
			]
			return {
				"ok": True,
				"action": "traffic",
				"city": city_name,
				"source": "google_maps",
				"report": "\n".join(report_lines),
				"maps_url": maps_url,
				"frontend_payload": {
					"type": "city_traffic",
					"title": f"Traffic in {city_name}",
					"summary": "Live congestion view opened in Google Maps.",
					"primary_url": maps_url,
					"actions": [
						{"label": "Open Live Traffic", "url": maps_url},
					],
				},
			}
		except Exception as exc:
			maps_city = (city or "your city").strip() or "your city"
			maps_url = "https://www.google.com/maps/search/" + urllib.parse.quote_plus(f"traffic in {maps_city}")
			try:
				self.open_url(maps_url)
			except Exception:
				pass
			return {
				"ok": False,
				"action": "traffic",
				"error": f"traffic unavailable: {exc}",
				"fallback_url": maps_url,
				"report": "I could not fetch live traffic metrics right now, but I opened Google Maps traffic.",
			}

	def get_route_traffic_report(self, origin: str, destination: str) -> Dict[str, Any]:
		"""Open route traffic in Google Maps directions mode for live congestion and ETA."""
		origin_clean = (origin or "").strip()
		destination_clean = (destination or "").strip()
		if not origin_clean or not destination_clean:
			return {
				"ok": False,
				"action": "route_traffic",
				"error": "missing origin or destination",
				"report": "Please say the route like: traffic from Mumbai to Pune.",
			}
		try:
			directions_url = (
				"https://www.google.com/maps/dir/?api=1"
				f"&origin={urllib.parse.quote_plus(origin_clean)}"
				f"&destination={urllib.parse.quote_plus(destination_clean)}"
				"&travelmode=driving"
				"&dir_action=navigate"
			)
			avoid_tolls_url = directions_url + "&avoid=tolls"
			avoid_highways_url = directions_url + "&avoid=highways"
			avoid_tolls_highways_url = directions_url + "&avoid=tolls|highways"
			traffic_layer_url = "https://www.google.com/maps/search/" + urllib.parse.quote_plus(f"traffic from {origin_clean} to {destination_clean}")
			self.open_url(directions_url)
			report_lines = [
				f"Opened live route traffic from {origin_clean} to {destination_clean} in Google Maps.",
				"Route options:",
				f"1. Fastest (default): {directions_url}",
				f"2. Avoid tolls: {avoid_tolls_url}",
				f"3. Avoid highways: {avoid_highways_url}",
				f"4. Avoid tolls + highways: {avoid_tolls_highways_url}",
				f"Traffic search view: {traffic_layer_url}",
			]
			return {
				"ok": True,
				"action": "route_traffic",
				"origin": origin_clean,
				"destination": destination_clean,
				"source": "google_maps",
				"directions_url": directions_url,
				"avoid_tolls_url": avoid_tolls_url,
				"avoid_highways_url": avoid_highways_url,
				"avoid_tolls_highways_url": avoid_tolls_highways_url,
				"traffic_url": traffic_layer_url,
				"report": "\n".join(report_lines),
				"frontend_payload": {
					"type": "route_traffic",
					"title": f"Traffic: {origin_clean} -> {destination_clean}",
					"summary": "Live ETA and route alternatives from Google Maps.",
					"primary_url": directions_url,
					"actions": [
						{"label": "Fastest", "url": directions_url},
						{"label": "Avoid Tolls", "url": avoid_tolls_url},
						{"label": "Avoid Highways", "url": avoid_highways_url},
						{"label": "Avoid Tolls + Highways", "url": avoid_tolls_highways_url},
						{"label": "Traffic Search", "url": traffic_layer_url},
					],
				},
			}
		except Exception as exc:
			return {
				"ok": False,
				"action": "route_traffic",
				"origin": origin_clean,
				"destination": destination_clean,
				"error": f"route traffic unavailable: {exc}",
				"report": "I could not open Google Maps route traffic right now.",
			}

	def execute_web_voice_command(self, command_text: str) -> Optional[Dict[str, Any]]:
		"""Parse and execute browser voice commands. Returns None if command is not web-related."""
		normalized = normalize_text(command_text)
		if not normalized:
			return None

		web_markers = [
			"youtube", "browser", "tab", "website", "site", "play", "pause", "resume", "mute", "unmute",
			"fullscreen", "search youtube", "next video", "previous video", "scroll", "go to",
		]
		if not any(marker in normalized for marker in web_markers):
			return None

		if re.search(r"\bplay\s+video\b", normalized):
			query = re.sub(r".*\bplay\s+video(?:\s+on\s+youtube)?(?:\s+(?:for|about))?\s*", "", command_text, flags=re.I).strip()
			if query:
				self._last_youtube_query = query
				return self._open_first_youtube_result(query)
			if self._last_youtube_query:
				return self._open_first_youtube_result(self._last_youtube_query)
			return self._play_first_result_from_current_youtube_page()

		if re.search(r"\b(close|stop)\s+(video|youtube|tab)\b", normalized):
			return self._press_key("ctrl+w")

		if re.search(r"\b(?:search(?:\s+(?:in|on))?\s+youtube|youtube\s+search)\b", normalized):
			query = re.sub(r".*(?:search(?:\s+(?:in|on))?\s+youtube(?:\s+for)?|youtube\s+search(?:\s+for)?)\s+", "", command_text, flags=re.I).strip()
			if not query:
				return {"ok": False, "error": "missing youtube search query"}
			self._last_youtube_query = query
			url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
			return {"ok": True, "action": "open_url", "url": url, "result": self.open_url(url)}

		if re.search(r"\b(open|go to|launch)\s+youtube\b", normalized):
			return {"ok": True, "action": "open_url", "url": "https://www.youtube.com", "result": self.open_url("https://www.youtube.com")}

		if re.search(r"\b(go to|open)\s+https?://", normalized):
			match = re.search(r"https?://\S+", command_text, flags=re.I)
			if match:
				url = match.group(0)
				return {"ok": True, "action": "open_url", "url": url, "result": self.open_url(url)}

		if re.search(r"\b(go to|open)\s+[a-z0-9\-]+\.[a-z]{2,}\b", normalized):
			match = re.search(r"\b([a-z0-9\-]+\.[a-z]{2,})\b", normalized)
			if match:
				domain = match.group(1)
				url = "https://" + domain
				return {"ok": True, "action": "open_url", "url": url, "result": self.open_url(url)}

		# YouTube keyboard controls (active tab should be YouTube).
		if re.search(r"\b(play|pause|resume|toggle play)\b", normalized):
			return self._press_key("k")
		if re.search(r"\b(next video|skip video)\b", normalized):
			return self._press_key("shift+n")
		if re.search(r"\b(previous video|last video)\b", normalized):
			return self._press_key("shift+p")
		if re.search(r"\b(mute|unmute)\b", normalized):
			return self._press_key("m")
		if "fullscreen" in normalized:
			return self._press_key("f")
		if "theater mode" in normalized:
			return self._press_key("t")
		if "captions" in normalized or "subtitles" in normalized:
			return self._press_key("c")
		if re.search(r"\b(forward|skip ahead|seek forward)\b", normalized):
			return self._press_key("l")
		if re.search(r"\b(back|rewind|seek back)\b", normalized):
			return self._press_key("j")
		if re.search(r"\b(volume up|increase volume|louder)\b", normalized):
			match = re.search(r"\b(volume up|increase volume|louder)\s+(\d{1,3})\b", normalized)
			if match:
				return self._press_key_repeated("up", max(1, min(20, int(match.group(2)) // 5)))
			return self._press_key_repeated("up", 4)
		if re.search(r"\b(volume down|decrease volume|quieter)\b", normalized):
			match = re.search(r"\b(volume down|decrease volume|quieter)\s+(\d{1,3})\b", normalized)
			if match:
				return self._press_key_repeated("down", max(1, min(20, int(match.group(2)) // 5)))
			return self._press_key_repeated("down", 4)
		if re.search(r"\b(volume off|no volume|mute all|silence)\b", normalized):
			return self._press_key("m")
		if re.search(r"\b(set|change|adjust)\s+volume\s+(?:to\s+)?(\d{1,3})\b", normalized):
			match = re.search(r"\b(set|change|adjust)\s+volume\s+(?:to\s+)?(\d{1,3})\b", normalized)
			if match:
				return self._set_youtube_volume_fast(int(match.group(2)))
		if re.search(r"\b(volume max|full volume|max volume|loudest)\b", normalized):
			return self._set_youtube_volume_fast(100)
		if re.search(r"\b(volume low|low volume|soft volume)\b", normalized):
			return self._set_youtube_volume_fast(20)

		# Generic browser actions.
		if re.search(r"\b(scroll (?:down|lower))\b", normalized):
			match = re.search(r"\bscroll (?:down|lower)(?:\s+(\d+))?\b", normalized)
			count = int(match.group(1)) if match and match.group(1) else 1
			return self._scroll_page("down", count)
		if re.search(r"\b(scroll (?:up|higher))\b", normalized):
			match = re.search(r"\bscroll (?:up|higher)(?:\s+(\d+))?\b", normalized)
			count = int(match.group(1)) if match and match.group(1) else 1
			return self._scroll_page("up", count)
		if re.search(r"\b(scroll to top|top of page)\b", normalized):
			return self._press_key("home")
		if re.search(r"\b(scroll to bottom|bottom of page)\b", normalized):
			return self._press_key("end")
		if re.search(r"\b(page down|next page)\b", normalized):
			return self._press_key("pagedown")
		if re.search(r"\b(page up|previous page)\b", normalized):
			return self._press_key("pageup")
		if re.search(r"\b(new tab)\b", normalized):
			return self._press_key("ctrl+t")
		if re.search(r"\b(close tab)\b", normalized):
			return self._press_key("ctrl+w")
		if re.search(r"\b(next tab)\b", normalized):
			return self._press_key("ctrl+tab")
		if re.search(r"\b(previous tab|last tab)\b", normalized):
			return self._press_key("ctrl+shift+tab")

		return {"ok": False, "error": "web command recognized but no executable action matched"}

	def system_stats(self) -> Dict[str, Any]:
		stats = {"platform": platform.platform(), "python": platform.python_version()}
		try:
			import psutil  # type: ignore
			stats.update({"cpu_percent": psutil.cpu_percent(interval=0.1), "memory_percent": psutil.virtual_memory().percent, "disk_percent": psutil.disk_usage(str(Path.home())).percent})
		except Exception:
			stats["resource_monitoring"] = "psutil not available"
		return stats

	def get_ip_report(self) -> Dict[str, Any]:
		local_ip = "unknown"
		public_ip = "unknown"
		try:
			sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			sock.connect(("8.8.8.8", 80))
			local_ip = sock.getsockname()[0]
			sock.close()
		except Exception:
			try:
				local_ip = socket.gethostbyname(socket.gethostname())
			except Exception:
				pass
		try:
			resp = requests.get("https://api.ipify.org?format=json", timeout=6)
			resp.raise_for_status()
			public_ip = str((resp.json() or {}).get("ip", "unknown"))
		except Exception:
			pass
		report = "\n".join([
			"IP address report:",
			f"Local IP: {local_ip}",
			f"Public IP: {public_ip}",
		])
		return {"ok": True, "action": "ip_report", "local_ip": local_ip, "public_ip": public_ip, "report": report}

	def get_gpu_report(self) -> Dict[str, Any]:
		try:
			query = [
				"nvidia-smi",
				"--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
				"--format=csv,noheader,nounits",
			]
			result = subprocess.run(query, capture_output=True, text=True, check=False, timeout=6)
			if result.returncode == 0 and (result.stdout or "").strip():
				lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
				report_lines = ["GPU report:"]
				gpus: List[Dict[str, Any]] = []
				for idx, line in enumerate(lines, start=1):
					parts = [part.strip() for part in line.split(",")]
					if len(parts) < 5:
						continue
					name, util, mem_used, mem_total, temp = parts[:5]
					report_lines.append(f"GPU {idx}: {name} | Usage {util}% | VRAM {mem_used}/{mem_total} MiB | Temp {temp}C")
					gpus.append({
						"index": idx,
						"name": name,
						"usage_percent": float(util),
						"memory_used_mib": float(mem_used),
						"memory_total_mib": float(mem_total),
						"temperature_c": float(temp),
					})
				if gpus:
					return {"ok": True, "action": "gpu_report", "source": "nvidia-smi", "gpus": gpus, "report": "\n".join(report_lines)}
		except Exception:
			pass

		# Fallback: adapter info without live usage.
		try:
			if platform.system().lower().startswith("win"):
				command = [
					"powershell",
					"-NoProfile",
					"-Command",
					"$g=Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name; $g -join '; '",
				]
				result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=6)
				names = (result.stdout or "").strip()
				if names:
					report = (
						"GPU report:\n"
						f"Detected GPU(s): {names}\n"
						"Live GPU usage is unavailable because NVIDIA telemetry (nvidia-smi) is not present."
					)
					return {"ok": True, "action": "gpu_report", "source": "win32_videocontroller", "gpus": names.split("; "), "report": report}
		except Exception:
			pass

		return {
			"ok": False,
			"action": "gpu_report",
			"error": "GPU telemetry unavailable",
			"report": "I could not read GPU usage on this system right now.",
		}

	def read_clipboard(self) -> str:
		try:
			import tkinter as tk  # type: ignore
			root = tk.Tk()
			root.withdraw()
			value = root.clipboard_get()
			root.destroy()
			return value
		except Exception:
			return ""

	def write_clipboard(self, text: str) -> bool:
		try:
			import tkinter as tk  # type: ignore
			root = tk.Tk()
			root.withdraw()
			root.clipboard_clear()
			root.clipboard_append(text)
			root.update()
			root.destroy()
			return True
		except Exception:
			return False


class VoicePipeline:
	"""Optional speech pipeline with offline-first fallbacks."""

	def __init__(self) -> None:
		self.listening = False
		self._interrupt_event = threading.Event()
		self._speaking_lock = threading.Lock()
		self._is_speaking = False
		self.last_error: str = ""
		self.wake_aliases = ["jarvis", "jervis", "jarviz", "jarvice", "jarves", "javis", "travis", "jarviss"]
		self.microphone_available = True
		self.microphone_error = ""
		self._consecutive_mic_failures = 0
		self._last_calibration_ts = 0.0
		self._wake_token_map = {
			"jarvice": "jarvis",
			"jarves": "jarvis",
			"jervis": "jarvis",
			"jarviz": "jarvis",
			"javis": "jarvis",
			"jarvish": "jarvis",
			"jarwis": "jarvis",
			"travis": "jarvis",
		}
		self._tts_rate = 165
		self._tts_volume = 1.0
		self._sapi_voice_name: Optional[str] = None
		self._sapi_voice_name_hi: Optional[str] = None
		self._pyttsx3_voice_id: Optional[str] = None
		self._pyttsx3_voice_id_hi: Optional[str] = None
		self._tts_voice_hints = [
			"aria",
			"jenny",
			"guy",
			"davis",
			"sara",
			"natasha",
			"neural",
			"natural",
			"zira",
			"hazel",
			"samantha",
		]
		self._tts_voice_hints_hi = ["heera", "kalpana", "hindi", "india", "hemant", "sangeeta", "female"]

	def transcribe_audio(self, audio_path: str) -> str:
		try:
			from faster_whisper import WhisperModel  # type: ignore
			model = WhisperModel(os.getenv("JARVIS_WHISPER_MODEL", "base"), device=os.getenv("JARVIS_WHISPER_DEVICE", "cpu"), compute_type=os.getenv("JARVIS_WHISPER_COMPUTE", "int8"))
			segments, _ = model.transcribe(audio_path)
			return " ".join(segment.text for segment in segments).strip()
		except Exception:
			try:
				import speech_recognition as sr  # type: ignore
				recognizer = sr.Recognizer()
				with sr.AudioFile(audio_path) as source:
					audio = recognizer.record(source)
				return recognizer.recognize_google(audio)
			except Exception:
				return ""

	def listen_once(self, timeout: int = 5, phrase_time_limit: int = 10, retries: int = 1, retry_delay: float = 0.15) -> str:
		import speech_recognition as sr  # type: ignore

		self.listening = True
		try:
			for attempt in range(max(0, retries) + 1):
				try:
					recognizer = sr.Recognizer()
					recognizer.energy_threshold = 240
					recognizer.dynamic_energy_threshold = True
					recognizer.pause_threshold = 0.6
					recognizer.non_speaking_duration = 0.35
					with sr.Microphone() as source:
						now_ts = time.time()
						if now_ts - self._last_calibration_ts > 45:
							recognizer.adjust_for_ambient_noise(source, duration=0.15)
							self._last_calibration_ts = now_ts
						audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
					transcript = ""
					last_recognition_error = ""
					for lang in ("en-IN", "en-US"):
						try:
							transcript = recognizer.recognize_google(audio, language=lang)
							if transcript:
								break
						except sr.UnknownValueError:
							last_recognition_error = "speech not understood"
							continue
						except Exception as rec_exc:
							last_recognition_error = str(rec_exc)
							break
					if transcript:
						self.last_error = ""
						self.microphone_error = ""
						self.microphone_available = True
						self._consecutive_mic_failures = 0
						return transcript
					self.last_error = last_recognition_error or "speech not understood"
					if attempt < retries and self.last_error == "speech not understood":
						time.sleep(max(0.0, retry_delay))
						continue
					return ""
				except KeyboardInterrupt:
					self.last_error = "microphone capture interrupted"
					return ""
				except Exception as exc:
					error_text = str(exc)
					self.last_error = error_text
					error_lower = error_text.lower()
					recoverable_markers = [
						"timed out while waiting for phrase to start",
						"audio data could not be read",
					]
					if attempt < retries and any(marker in error_lower for marker in recoverable_markers):
						time.sleep(max(0.0, retry_delay))
						continue
					self._consecutive_mic_failures += 1
					hard_error_markers = [
						"NoneType' object has no attribute 'close'",
						"No Default Input Device Available",
						"Invalid input device",
						"PyAudio",
						"WinError",
					]
					if any(marker.lower() in error_text.lower() for marker in hard_error_markers) or self._consecutive_mic_failures >= 3:
						self.microphone_available = False
						self.microphone_error = error_text
					return ""
			return ""
		finally:
			self.listening = False

	def reset_microphone_state(self) -> None:
		self.microphone_available = True
		self.microphone_error = ""
		self.last_error = ""
		self._consecutive_mic_failures = 0

	@property
	def is_speaking(self) -> bool:
		with self._speaking_lock:
			return self._is_speaking

	def request_interrupt(self) -> None:
		self._interrupt_event.set()

	def clear_interrupt(self) -> None:
		self._interrupt_event.clear()

	def _prepare_speech_text(self, text: str) -> str:
		cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
		cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
		cleaned = re.sub(r"^\s*#{1,6}\s+", "", cleaned, flags=re.M)
		cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned, flags=re.M)
		cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.M)
		cleaned = cleaned.replace("\r", " ").replace("\n", " ")
		cleaned = re.sub(r"\s+", " ", cleaned).strip()
		return cleaned

	def _split_speech_chunks(self, text: str) -> List[str]:
		if not text.strip():
			return []
		sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
		if not sentences:
			sentences = [text.strip()]
		chunks: List[str] = []
		buffer = ""
		max_len = 260
		for sentence in sentences:
			candidate = f"{buffer} {sentence}".strip() if buffer else sentence
			if len(candidate) <= max_len:
				buffer = candidate
				continue
			if buffer:
				chunks.append(buffer)
				buffer = ""
			if len(sentence) <= max_len:
				buffer = sentence
			else:
				words = sentence.split()
				part = ""
				for word in words:
					word_candidate = f"{part} {word}".strip() if part else word
					if len(word_candidate) <= max_len:
						part = word_candidate
					else:
						if part:
							chunks.append(part)
						part = word
				if part:
					buffer = part
		if buffer:
			chunks.append(buffer)
		return chunks

	def _looks_hindi_text(self, text: str) -> bool:
		if re.search(r"[\u0900-\u097F]", text):
			return True
		tokens = set(tokenise(text))
		hindi_roman_markers = {
			"kya", "kaise", "kyu", "kyun", "nahi", "nahin", "haan", "ha", "achha", "accha",
			"thik", "theek", "jaldi", "abhi", "chahiye", "kar", "karo", "karta", "kr", "krna",
			"bolo", "batao", "samjha", "samjhao", "mera", "meri", "mujhe", "tum", "aap", "bhai",
			"yaar", "sahi", "galat", "thoda", "zyada", "hai", "ho", "main", "tumhara", "namaste",
		}
		return len(tokens & hindi_roman_markers) >= 2

	def _roman_hindi_to_devanagari(self, text: str) -> str:
		if re.search(r"[\u0900-\u097F]", text):
			return text
		mapping = {
			"namaste": "नमस्ते", "main": "मैं", "mera": "मेरा", "meri": "मेरी", "mujhe": "मुझे",
			"tum": "तुम", "aap": "आप", "hai": "है", "ho": "हो", "haan": "हाँ", "ha": "हाँ",
			"nahi": "नहीं", "nahin": "नहीं", "kya": "क्या", "kaise": "कैसे", "kyu": "क्यों",
			"kyun": "क्यों", "kar": "कर", "karo": "करो", "kr": "कर", "krna": "करना",
			"chahiye": "चाहिए", "abhi": "अभी", "jaldi": "जल्दी", "achha": "अच्छा", "accha": "अच्छा",
			"thik": "ठीक", "theek": "ठीक", "bolo": "बोलो", "batao": "बताओ", "samjhao": "समझाओ",
			"bhai": "भाई", "yaar": "यार", "sahi": "सही", "galat": "गलत", "thoda": "थोड़ा", "zyada": "ज़्यादा",
		}
		replaced = 0

		def _replace(match: re.Match[str]) -> str:
			nonlocal replaced
			word = match.group(0)
			repl = mapping.get(word.lower())
			if repl:
				replaced += 1
				return repl
			return word

		converted = re.sub(r"[A-Za-z']+", _replace, text)
		return converted if replaced >= 2 else text

	def _voice_hints_for_text(self, text: str) -> List[str]:
		return self._tts_voice_hints_hi if self._looks_hindi_text(text) else self._tts_voice_hints

	def _speak_hindi_neural(self, text: str) -> bool:
		"""Use Edge neural Hindi TTS when available for higher fluency."""
		voice_name = os.getenv("JARVIS_HINDI_NEURAL_VOICE", "hi-IN-SwaraNeural").strip() or "hi-IN-SwaraNeural"
		rate = os.getenv("JARVIS_HINDI_NEURAL_RATE", "+0%").strip() or "+0%"
		volume = os.getenv("JARVIS_HINDI_NEURAL_VOLUME", "+0%").strip() or "+0%"
		return self._speak_neural_with_voice(text, voice_name=voice_name, rate=rate, volume=volume)

	def _speak_neural_with_voice(self, text: str, voice_name: str, rate: str = "+0%", volume: str = "+0%") -> bool:
		"""Use Edge neural TTS with an explicit voice name."""
		if not text.strip():
			return False
		try:
			import edge_tts  # type: ignore
			import pygame  # type: ignore
		except Exception:
			# Try loading from project-local venv site-packages.
			project_dir = Path(__file__).resolve().parent
			venv_site = project_dir / ".jarvis-venv" / "Lib" / "site-packages"
			if venv_site.exists() and str(venv_site) not in sys.path:
				sys.path.append(str(venv_site))
			try:
				import edge_tts  # type: ignore
				import pygame  # type: ignore
			except Exception:
				return False

		temp_path = ""
		try:
			with tempfile.NamedTemporaryFile(prefix="jarvis_hi_", suffix=".mp3", delete=False) as tmp:
				temp_path = tmp.name

			async def _render() -> None:
				communicate = edge_tts.Communicate(text=text, voice=voice_name, rate=rate, volume=volume)
				await communicate.save(temp_path)

			asyncio.run(_render())
			pygame.mixer.init()
			pygame.mixer.music.load(temp_path)
			pygame.mixer.music.play()
			while pygame.mixer.music.get_busy():
				if self._interrupt_event.is_set():
					pygame.mixer.music.stop()
					return False
				time.sleep(0.05)
			pygame.mixer.music.stop()
			return True
		except Exception:
			return False
		finally:
			try:
				import pygame  # type: ignore
				pygame.mixer.quit()
			except Exception:
				pass
			if temp_path:
				try:
					Path(temp_path).unlink(missing_ok=True)
				except Exception:
					pass

	def _build_sapi_ssml(self, text: str, xml_lang: str = "en-US") -> str:
		xml_text = html.escape(text, quote=False)
		rate = "-10%" if xml_lang == "hi-IN" else "-8%"
		return (
			f"<speak version='1.0' xml:lang='{xml_lang}'>"
			f"<prosody rate='{rate}' pitch='+0%'>"
			f"{xml_text}"
			"</prosody>"
			"</speak>"
		)

	def _speak_windows_sapi(self, text: str) -> bool:
		try:
			hindi_context = self._looks_hindi_text(text)
			speech_text = self._roman_hindi_to_devanagari(text) if hindi_context else text
			xml_lang = "hi-IN" if hindi_context else "en-US"
			rate = -3 if hindi_context else -2
			escaped = speech_text.replace("'", "''")
			escaped_ssml = self._build_sapi_ssml(speech_text, xml_lang=xml_lang).replace("'", "''")
			voice_name = (self._sapi_voice_name_hi if hindi_context else self._sapi_voice_name) or ""
			escaped_voice_name = voice_name.replace("'", "''")
			voice_assign = ""
			if voice_name:
				voice_assign = f"$synth.SelectVoice('{escaped_voice_name}'); "
			voice_hint_regex = "|".join(self._voice_hints_for_text(speech_text))
			fallback_regex = "heera|kalpana|hindi|india|zira|hazel|samantha|david|mark" if hindi_context else "zira|hazel|samantha|david|mark"
			command = [
				"powershell",
				"-NoProfile",
				"-Command",
				f"Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
				f"if (-not '{escaped_voice_name}') {{ "
				f"$v = $synth.GetInstalledVoices() | ForEach-Object {{$_.VoiceInfo}} | Where-Object {{$_.Name -match '{voice_hint_regex}' -or $_.Description -match 'natural|neural'}} | Select-Object -First 1; "
				f"if (-not $v) {{ $v = $synth.GetInstalledVoices() | ForEach-Object {{$_.VoiceInfo}} | Where-Object {{$_.Gender -eq 'Female' -or $_.Name -match '{fallback_regex}'}} | Select-Object -First 1 }} "
				f"if ($v) {{$synth.SelectVoice($v.Name)}} "
				f"}} else {{ {voice_assign} }} "
				f"$synth.Rate = {rate}; $synth.Volume = 100; "
				f"try {{ $synth.SpeakSsml('{escaped_ssml}') }} catch {{ $synth.Speak('{escaped}') }}",
			]
			result = subprocess.run(command, capture_output=True, text=True, check=False)
			if result.returncode == 0 and hindi_context and not self._sapi_voice_name_hi:
				self._sapi_voice_name_hi = self._detect_windows_preferred_voice_name(self._tts_voice_hints_hi, fallback_regex="heera|kalpana|hindi|india|zira|hazel|samantha|david|mark")
			if result.returncode == 0 and (not hindi_context) and not self._sapi_voice_name:
				# Cache a preferred voice name for next calls.
				self._sapi_voice_name = self._detect_windows_preferred_voice_name()
			return result.returncode == 0
		except Exception:
			return False

	def _detect_windows_preferred_voice_name(self, hints: Optional[List[str]] = None, fallback_regex: str = "zira|hazel|samantha|david|mark") -> Optional[str]:
		try:
			voice_hint_regex = "|".join(hints or self._tts_voice_hints)
			command = [
				"powershell",
				"-NoProfile",
				"-Command",
				"Add-Type -AssemblyName System.Speech; "
				"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
				"$v = $synth.GetInstalledVoices() | ForEach-Object {$_.VoiceInfo} | "
				f"Where-Object {{$_.Name -match '{voice_hint_regex}' -or $_.Description -match 'natural|neural'}} | Select-Object -First 1; "
				f"if (-not $v) {{ $v = $synth.GetInstalledVoices() | ForEach-Object {{$_.VoiceInfo}} | Where-Object {{$_.Gender -eq 'Female' -or $_.Name -match '{fallback_regex}'}} | Select-Object -First 1 }} "
				"if ($v) { Write-Output $v.Name }",
			]
			result = subprocess.run(command, capture_output=True, text=True, check=False)
			voice = (result.stdout or "").strip()
			return voice or None
		except Exception:
			return None

	def _configure_pyttsx3_engine(self, engine: Any, speech_text: str = "") -> None:
		try:
			hindi_context = self._looks_hindi_text(speech_text)
			voices = engine.getProperty("voices") or []
			cached_id = self._pyttsx3_voice_id_hi if hindi_context else self._pyttsx3_voice_id
			if cached_id:
				engine.setProperty("voice", cached_id)
			else:
				female_markers = ["female", "heera", "kalpana", "hindi", "india", "hemant"] if hindi_context else ["female", "aria", "jenny", "guy", "davis", "natasha", "zira", "hazel", "samantha", "david", "mark", "kalpana"]
				selected_id: Optional[str] = None
				for voice in voices:
					blob = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
					if any(marker in blob for marker in female_markers):
						selected_id = getattr(voice, "id", None)
						break
				if selected_id:
					engine.setProperty("voice", selected_id)
					if hindi_context:
						self._pyttsx3_voice_id_hi = selected_id
					else:
						self._pyttsx3_voice_id = selected_id
			engine.setProperty("volume", self._tts_volume)
			engine.setProperty("rate", 155 if hindi_context else self._tts_rate)
		except Exception:
			pass

	def speak(self, text: str) -> bool:
		text = self._prepare_speech_text((text or "").strip())
		if not text:
			return False
		hindi_context = self._looks_hindi_text(text)
		speech_text = self._roman_hindi_to_devanagari(text) if hindi_context else text
		self.clear_interrupt()
		with self._speaking_lock:
			self._is_speaking = True
		completed = True
		try:
			spoken = False
			if hindi_context:
				spoken = self._speak_hindi_neural(speech_text)
			if not spoken and platform.system().lower().startswith("win"):
				if self._sapi_voice_name is None:
					self._sapi_voice_name = self._detect_windows_preferred_voice_name()
				spoken = self._speak_windows_sapi(speech_text)
			if not spoken:
				import pyttsx3  # type: ignore
				engine = pyttsx3.init()
				self._configure_pyttsx3_engine(engine, speech_text=speech_text)
				if self._interrupt_event.is_set():
					completed = False
					engine.stop()
				else:
					engine.say(speech_text)
					engine.runAndWait()
		except Exception:
			try:
				import pyttsx3  # type: ignore
				for chunk in self._split_speech_chunks(speech_text):
					if self._interrupt_event.is_set():
						completed = False
						break
					engine = pyttsx3.init()
					self._configure_pyttsx3_engine(engine, speech_text=chunk)
					engine.say(chunk)
					engine.runAndWait()
					engine.stop()
			except Exception:
				print(f"Jarvis (voice): {speech_text}")
		finally:
			with self._speaking_lock:
				self._is_speaking = False
			self.clear_interrupt()
		return completed

	def wake_word_detected(self, transcript: str) -> bool:
		return self.extract_command_after_wake(transcript) is not None

	def _canonical_wake_token(self, token: str, previous_token: str = "") -> str:
		if token == "service" and previous_token in {"hey", "hi", "ok", "okay"}:
			return "jarvis"
		return self._wake_token_map.get(token, token)

	def extract_command_after_wake(self, transcript: str) -> Optional[str]:
		normalized = normalize_text(transcript)
		if not normalized:
			return None
		match = re.match(r"^(?:hey\s+|hi\s+|ok\s+|okay\s+)?(?:jarviss|jarvice|jarvis|jervis|jarviz|jarves|javis|travis)\b[,:\-\s]*(.*)$", normalized, flags=re.I)
		if match:
			remainder = match.group(1).strip()
			return remainder or ""
		cleaned = re.sub(r"[^a-z0-9\s]", " ", normalized)
		tokens = [token for token in cleaned.split() if token]
		canonical_tokens: List[str] = []
		for idx, token in enumerate(tokens):
			prev = tokens[idx - 1] if idx > 0 else ""
			canonical_tokens.append(self._canonical_wake_token(token, prev))

		for idx, token in enumerate(canonical_tokens[:10]):
			if token == "jarvis":
				remainder = " ".join(tokens[idx + 1:]).strip()
				return remainder or ""

		for idx, token in enumerate(canonical_tokens[:10]):
			prev = canonical_tokens[idx - 1] if idx > 0 else ""
			threshold = 0.78
			if prev in {"hey", "hi", "ok", "okay"}:
				threshold = 0.62
			if any(difflib.SequenceMatcher(None, token, alias).ratio() >= threshold for alias in self.wake_aliases):
				remainder = " ".join(tokens[idx + 1:]).strip()
				return remainder or ""

		prefix = " ".join(tokens[:3])
		if prefix:
			for candidate in ("hey jarvis", "hi jarvis", "ok jarvis", "okay jarvis"):
				if difflib.SequenceMatcher(None, prefix, candidate).ratio() >= 0.64:
					remainder_tokens = tokens[2:] if len(tokens) > 2 else []
					remainder = " ".join(remainder_tokens).strip()
					return remainder or ""
		return None


class HudOverlay:
	"""Dashboard-style Jarvis UI with animated core, waveform, and readable content panes."""

	def __init__(self) -> None:
		self._root = None
		self._title_label = None
		self._status_label = None
		self._subtitle_label = None
		self._telemetry_label = None
		self._wave_canvas = None
		self._core_canvas = None
		self._core_items: List[int] = []
		self._wave_bars: List[int] = []
		self._left_buttons: List[Any] = []
		self._right_badges: List[Any] = []
		self._transcript_value = None
		self._response_value = None
		self._command_entry = None
		self._send_button = None
		self._command_callback = None
		self._mode = "idle"
		self._wave_phase = 0.0
		self._core_phase = 0.0
		self._last_text = "JARVIS READY"
		self._last_transcript = ""
		self._last_response = ""
		self._learning_info = "Learning: ON | waiting for interaction"
		self._target_fps = 24
		self._last_render_ts = 0.0
		self._last_text_render_ts = 0.0
		self._text_render_interval = 0.08
		self._wave_bar_count = 30
		self._wave_bar_spacing = 30
		self._wave_base_x = 16
		self._rendered_transcript = ""
		self._rendered_response = ""
		self._tick_counter = 0

	def set_command_callback(self, callback: Any) -> None:
		self._command_callback = callback

	def _submit_ui_command(self) -> None:
		if self._command_entry is None or self._command_callback is None:
			return
		try:
			value = self._command_entry.get().strip()
			if not value:
				return
			self._command_entry.delete(0, "end")
			self._command_callback(value)
		except Exception:
			pass

	def _trigger_quick_action(self, label: str) -> None:
		if self._command_callback is None:
			return
		quick_actions = {
			"Chrome": "open chrome",
			"Research": "research top AI updates today",
			"Memory": "what do you remember about me",
			"System": "system stats",
			"Briefing": "daily briefing",
			"Settings": "show pipeline status",
		}
		command = quick_actions.get(label)
		if command:
			try:
				self._command_callback(command)
			except Exception:
				pass

	def _mode_color(self) -> str:
		if self._mode == "listening":
			return "#22d3ee"
		if self._mode == "processing":
			return "#f59e0b"
		if self._mode == "speaking":
			return "#fb7185"
		if self._mode == "active":
			return "#60a5fa"
		return "#64748b"

	def _mode_accent(self) -> str:
		if self._mode == "speaking":
			return "#fb7185"
		if self._mode == "processing":
			return "#f59e0b"
		if self._mode == "listening":
			return "#22d3ee"
		return "#38bdf8"

	def _ensure_ui(self) -> bool:
		try:
			import tkinter as tk  # type: ignore
			from tkinter import scrolledtext  # type: ignore
			if self._root is not None:
				try:
					if int(self._root.winfo_exists()) == 0:
						self._root = None
						self._title_label = None
						self._status_label = None
						self._subtitle_label = None
						self._telemetry_label = None
						self._wave_canvas = None
						self._core_canvas = None
						self._core_items = []
						self._wave_bars = []
						self._left_buttons = []
						self._right_badges = []
						self._transcript_value = None
						self._response_value = None
						self._command_entry = None
						self._send_button = None
				except Exception:
					self._root = None
					self._title_label = None
					self._status_label = None
					self._subtitle_label = None
					self._telemetry_label = None
					self._wave_canvas = None
					self._core_canvas = None
					self._core_items = []
					self._wave_bars = []
					self._left_buttons = []
					self._right_badges = []
					self._transcript_value = None
					self._response_value = None
					self._command_entry = None
					self._send_button = None

			if self._root is None:
				self._root = tk.Tk()
				self._root.title(f"{APP_NAME} Tactical Interface")
				self._root.attributes("-topmost", True)
				self._root.attributes("-alpha", 0.97)
				self._root.configure(bg="#050814")
				self._root.geometry("1360x820+20+20")

				container = tk.Frame(self._root, bg="#050814", padx=16, pady=16)
				container.pack(fill="both", expand=True)

				header = tk.Frame(container, bg="#0b1220", highlightthickness=1, highlightbackground="#24324a")
				header.pack(fill="x")
				self._title_label = tk.Label(header, text="JARVIS // TACTICAL INTERFACE", bg="#0b1220", fg="#dbeafe", font=("Consolas", 19, "bold"), anchor="w")
				self._title_label.pack(side="left", padx=16, pady=12)
				self._status_label = tk.Label(header, text=self._last_text, bg="#0b1220", fg="#22d3ee", font=("Consolas", 11, "bold"), anchor="e")
				self._status_label.pack(side="right", padx=16)

				self._subtitle_label = tk.Label(container, text="Voice-first control room with live transcript, response, and system telemetry.", bg="#050814", fg="#94a3b8", font=("Consolas", 10), anchor="w")
				self._subtitle_label.pack(fill="x", pady=(6, 10))

				main = tk.Frame(container, bg="#050814")
				main.pack(fill="both", expand=True)

				left = tk.Frame(main, bg="#0b1220", highlightthickness=1, highlightbackground="#24324a", width=250)
				left.pack(side="left", fill="y", padx=(0, 12))
				left.pack_propagate(False)
				dock_title = tk.Label(left, text="SYSTEM DOCK", bg="#0b1220", fg="#93c5fd", font=("Consolas", 11, "bold"), anchor="w")
				dock_title.pack(fill="x", padx=12, pady=(12, 8))
				for label in ["ONLINE", "AUDIO", "MEMORY", "RESEARCH"]:
					chip = tk.Label(left, text=label, bg="#111827", fg="#38bdf8", font=("Consolas", 9, "bold"), anchor="center", padx=10, pady=5)
					chip.pack(fill="x", padx=12, pady=3)
				for label in ["Chrome", "Research", "Memory", "System", "Briefing", "Settings"]:
					btn = tk.Label(left, text=label, bg="#111827", fg="#e5e7eb", font=("Consolas", 10, "bold"), anchor="w", padx=12, pady=10)
					btn.pack(fill="x", padx=12, pady=5)
					btn.bind("<Button-1>", lambda _evt, name=label: self._trigger_quick_action(name))
					btn.bind("<Enter>", lambda _evt, w=btn: w.config(bg="#1f2937"))
					btn.bind("<Leave>", lambda _evt, w=btn: w.config(bg="#111827"))
					self._left_buttons.append(btn)

				self._core_canvas = tk.Canvas(left, width=220, height=220, bg="#0b1220", highlightthickness=0)
				self._core_canvas.pack(pady=(10, 12))
				self._core_items = [
					self._core_canvas.create_oval(22, 22, 198, 198, outline="#38bdf8", width=4),
					self._core_canvas.create_oval(46, 46, 174, 174, outline="#f59e0b", width=3),
					self._core_canvas.create_oval(88, 88, 132, 132, fill="#22d3ee", outline=""),
					self._core_canvas.create_line(110, 18, 110, 202, fill="#1d4ed8", width=1),
					self._core_canvas.create_line(18, 110, 202, 110, fill="#1d4ed8", width=1),
					self._core_canvas.create_text(110, 206, text="NEURAL CORE", fill="#94a3b8", font=("Consolas", 9, "bold")),
				]

				center = tk.Frame(main, bg="#050814")
				center.pack(side="left", fill="both", expand=True, padx=(0, 12))
				self._wave_canvas = tk.Canvas(center, height=110, bg="#0b1220", highlightthickness=1, highlightbackground="#24324a")
				self._wave_canvas.pack(fill="x", pady=(0, 12))
				for i in range(self._wave_bar_count):
					x1 = self._wave_base_x + i * self._wave_bar_spacing
					bar = self._wave_canvas.create_rectangle(x1, 54, x1 + 10, 74, fill="#38bdf8", outline="")
					self._wave_bars.append(bar)

				transcript_frame = tk.Frame(center, bg="#0b1220", highlightthickness=1, highlightbackground="#24324a")
				transcript_frame.pack(fill="x", pady=(0, 12))
				label = tk.Label(transcript_frame, text="INPUT TRANSCRIPT", bg="#0b1220", fg="#93c5fd", font=("Consolas", 11, "bold"), anchor="w")
				label.pack(fill="x", padx=12, pady=(10, 0))
				self._transcript_value = scrolledtext.ScrolledText(transcript_frame, height=6, wrap="word", bg="#0f172a", fg="#e5e7eb", insertbackground="#e5e7eb", font=("Consolas", 11), relief="flat", borderwidth=0)
				self._transcript_value.pack(fill="x", padx=12, pady=(8, 12))

				response_frame = tk.Frame(center, bg="#0b1220", highlightthickness=1, highlightbackground="#24324a")
				response_frame.pack(fill="both", expand=True)
				r_label = tk.Label(response_frame, text="SPOKEN RESPONSE", bg="#0b1220", fg="#fbbf24", font=("Consolas", 11, "bold"), anchor="w")
				r_label.pack(fill="x", padx=12, pady=(10, 0))
				self._response_value = scrolledtext.ScrolledText(response_frame, wrap="word", bg="#0f172a", fg="#f8fafc", insertbackground="#f8fafc", font=("Consolas", 11), relief="flat", borderwidth=0)
				self._response_value.pack(fill="both", expand=True, padx=12, pady=(8, 12))

				input_frame = tk.Frame(center, bg="#0b1220", highlightthickness=1, highlightbackground="#24324a")
				input_frame.pack(fill="x", pady=(12, 0))
				input_label = tk.Label(input_frame, text="COMMAND INPUT", bg="#0b1220", fg="#93c5fd", font=("Consolas", 10, "bold"), anchor="w")
				input_label.pack(fill="x", padx=12, pady=(8, 4))
				input_row = tk.Frame(input_frame, bg="#0b1220")
				input_row.pack(fill="x", padx=12, pady=(0, 10))
				self._command_entry = tk.Entry(input_row, bg="#0f172a", fg="#e5e7eb", insertbackground="#e5e7eb", relief="flat", font=("Consolas", 11))
				self._command_entry.pack(side="left", fill="x", expand=True, ipady=6)
				self._command_entry.bind("<Return>", lambda _evt: self._submit_ui_command())
				self._send_button = tk.Button(input_row, text="Send", command=self._submit_ui_command, bg="#1d4ed8", fg="#e5e7eb", activebackground="#1e40af", activeforeground="#ffffff", relief="flat", font=("Consolas", 10, "bold"), padx=14, pady=6)
				self._send_button.pack(side="left", padx=(10, 0))

				right = tk.Frame(main, bg="#0b1220", highlightthickness=1, highlightbackground="#24324a", width=250)
				right.pack(side="left", fill="y")
				right.pack_propagate(False)
				right_title = tk.Label(right, text="STATUS", bg="#0b1220", fg="#93c5fd", font=("Consolas", 11, "bold"), anchor="w")
				right_title.pack(fill="x", padx=12, pady=(12, 8))
				for label in ["Voice Online", "Model: NVIDIA", "Mode: Active", "Mic: Ready", "Hotkey: Ctrl+Space"]:
					badge = tk.Label(right, text=label, bg="#111827", fg="#e5e7eb", font=("Consolas", 10), anchor="w", padx=10, pady=8)
					badge.pack(fill="x", padx=12, pady=5)
					self._right_badges.append(badge)

				telemetry = tk.Frame(right, bg="#111827", highlightthickness=1, highlightbackground="#24324a")
				telemetry.pack(fill="x", padx=12, pady=(12, 0))
				self._telemetry_label = tk.Label(telemetry, text="SYSTEM TELEMETRY", bg="#111827", fg="#f59e0b", font=("Consolas", 10, "bold"), anchor="w")
				self._telemetry_label.pack(fill="x", padx=10, pady=(10, 4))
				self._telemetry_text = tk.Label(telemetry, text="Listening for wake phrase.", bg="#111827", fg="#cbd5e1", font=("Consolas", 10), justify="left", anchor="w", wraplength=200)
				self._telemetry_text.pack(fill="x", padx=10, pady=(0, 10))


				if self._transcript_value is not None:
					self._transcript_value.insert("1.0", self._last_transcript)
				if self._response_value is not None:
					self._response_value.insert("1.0", self._last_response)
				if self._command_entry is not None:
					self._command_entry.focus_set()
			return True
		except Exception:
			return False

	def set_mode(self, mode: str, text: Optional[str] = None) -> bool:
		changed = False
		if mode:
			if mode != self._mode:
				self._mode = mode
				changed = True
		if text:
			if text != self._last_text:
				self._last_text = text
				changed = True
		if not self._ensure_ui():
			return False
		if changed:
			self._tick(force=True)
		else:
			self._tick(force=False)
		return True

	def set_transcript(self, text: str) -> bool:
		self._last_transcript = text or ""
		if not self._ensure_ui():
			return False
		try:
			now = time.time()
			if self._transcript_value is not None and self._last_transcript != self._rendered_transcript:
				if (now - self._last_text_render_ts) < self._text_render_interval and len(self._last_transcript) < len(self._rendered_transcript):
					self._tick(force=False)
					return True
				self._transcript_value.delete("1.0", "end")
				self._transcript_value.insert("1.0", self._last_transcript)
				self._rendered_transcript = self._last_transcript
				self._last_text_render_ts = now
			self._tick(force=False)
			return True
		except Exception:
			return False

	def set_response(self, text: str) -> bool:
		self._last_response = text or ""
		if not self._ensure_ui():
			return False
		try:
			if self._response_value is not None and self._last_response != self._rendered_response:
				self._response_value.delete("1.0", "end")
				self._response_value.insert("1.0", self._last_response)
				self._rendered_response = self._last_response
			self._tick(force=False)
			return True
		except Exception:
			return False

	def set_learning_info(self, text: str) -> bool:
		self._learning_info = (text or "").strip() or "Learning: ON | waiting for interaction"
		if not self._ensure_ui():
			return False
		try:
			self._tick(force=False)
			return True
		except Exception:
			return False

	def show(self, text: str) -> bool:
		return self.set_mode("active", text)

	def _tick(self, force: bool = False) -> None:
		if self._root is None:
			return
		try:
			self._tick_counter += 1
			now = time.time()
			interval = 1.0 / max(1, self._target_fps)
			if not force and (now - self._last_render_ts) < interval:
				return
			self._last_render_ts = now

			accent = self._mode_accent()
			mode_color = self._mode_color()
			if self._status_label is not None:
				self._status_label.config(text=self._last_text, fg=mode_color)
			if self._telemetry_text is not None:
				base_telemetry = {"idle": "Awaiting input.", "active": "Wake phrase armed and waiting.", "listening": "Microphone capture in progress.", "processing": "Thinking and routing the request.", "speaking": "Speaking the response now."}.get(self._mode, "Ready.")
				telemetry = base_telemetry + "\n\n" + self._learning_info
				self._telemetry_text.config(text=telemetry)
			for badge in self._right_badges:
				try:
					badge.config(fg=mode_color)
				except Exception:
					pass
			if self._telemetry_label is not None:
				self._telemetry_label.config(fg=accent)
			if self._title_label is not None:
				self._title_label.config(fg=accent)

			if self._mode == "listening":
				amplitude = 28
			elif self._mode == "speaking":
				amplitude = 34
			elif self._mode == "processing":
				amplitude = 16
			elif self._mode == "active":
				amplitude = 10
			else:
				amplitude = 5

			animate_full = force or self._mode in {"listening", "speaking", "processing"} or (self._tick_counter % 2 == 0)

			if self._wave_canvas is not None and animate_full:
				self._wave_phase += 0.2
				base_y = 74
				for i, bar in enumerate(self._wave_bars):
					height = 6 + int(amplitude * (0.35 + abs(math.sin(self._wave_phase + i * 0.42))))
					x1 = self._wave_base_x + i * self._wave_bar_spacing
					x2 = x1 + 10
					y1 = base_y - height
					self._wave_canvas.coords(bar, x1, y1, x2, base_y)
					self._wave_canvas.itemconfig(bar, fill=accent)

			if self._core_canvas is not None and self._core_items and animate_full:
				self._core_phase += 0.16
				outer = 78 + int(6 * (0.5 + abs(math.sin(self._core_phase))))
				mid = 52 + int(5 * (0.5 + abs(math.sin(self._core_phase + 0.7))))
				inner = 18 + int(4 * (0.5 + abs(math.sin(self._core_phase + 1.2))))
				self._core_canvas.coords(self._core_items[0], 106 - outer, 106 - outer, 106 + outer, 106 + outer)
				self._core_canvas.coords(self._core_items[1], 106 - mid, 106 - mid, 106 + mid, 106 + mid)
				self._core_canvas.coords(self._core_items[2], 106 - inner, 106 - inner, 106 + inner, 106 + inner)
				self._core_canvas.itemconfig(self._core_items[0], outline=accent)
				self._core_canvas.itemconfig(self._core_items[1], outline="#f59e0b" if self._mode != "speaking" else "#fb7185")
				self._core_canvas.itemconfig(self._core_items[2], fill=accent)

			if force:
				self._root.update_idletasks()
			self._root.update()
		except Exception:
			self._root = None
			self._title_label = None
			self._status_label = None
			self._subtitle_label = None
			self._telemetry_label = None
			self._wave_canvas = None
			self._core_canvas = None
			self._core_items = []
			self._wave_bars = []
			self._left_buttons = []
			self._right_badges = []
			self._transcript_value = None
			self._response_value = None
			self._command_entry = None
			self._send_button = None

	def set_gesture(self, gesture: str, confidence: float = 0.0) -> bool:
		"""Update gesture display."""
		if not self._ensure_ui():
			return False
		try:
			if self._gesture_label:
				text = f"GESTURE: {gesture.upper()} ({confidence:.0%})" if gesture != "idle" else "GESTURE: IDLE"
				if text != self._last_gesture_text:
					self._gesture_label.config(text=text, fg="#38bdf8" if gesture == "idle" else "#22d3ee")
					self._last_gesture_text = text
			self._tick(force=False)
			return True
		except Exception:
			return False

	def set_emotion(self, emotion: str, confidence: float = 0.0) -> bool:
		"""Update emotion display."""
		if not self._ensure_ui():
			return False
		try:
			if self._emotion_label:
				text = "EMOTION: OFF"
				if text != self._last_emotion_text:
					self._emotion_label.config(text=text, fg="#64748b")
					self._last_emotion_text = text
				self._tick(force=False)
				return True
		except Exception:
			pass
		return False

	def set_auth_status(self, authenticated: bool) -> bool:
		"""Update authentication status display."""
		if not self._ensure_ui():
			return False
		try:
			if self._auth_status_label:
				status_text = "✓ AUTHENTICATED" if authenticated else "FACE AUTH LOCKED"
				status_color = "#22c55e" if authenticated else "#ef4444"
				if status_text != self._last_auth_text:
					self._auth_status_label.config(text=status_text, fg=status_color)
					self._last_auth_text = status_text
			self._tick(force=False)
			return True
		except Exception:
			return False

	def close(self) -> None:
		try:
			if self._root is not None:
				self._root.destroy()
		except Exception:
			pass
		self._root = None
		self._title_label = None
		self._status_label = None
		self._subtitle_label = None
		self._telemetry_label = None
		self._wave_canvas = None
		self._core_canvas = None
		self._core_items = []
		self._wave_bars = []
		self._left_buttons = []
		self._right_badges = []
		self._transcript_value = None
		self._response_value = None
		self._command_entry = None
		self._send_button = None


class Brain:
	"""Main orchestration layer for reasoning, memory, and response planning."""

	def __init__(self, config: JarvisConfig, memory: MemoryStore):
		self.config = config
		self.memory = memory
		self.user_model = UserModel(memory)
		self.router = IntentRouter()
		self.intent_classifier = IntentClassifierLLM()
		self.persona = PersonaEngine()
		self.dialogue = DialogueTuner()
		self.research = ResearchService(LLMOrchestrator(config))
		self.system = SystemController()
		self.reasoning_mode = "balanced"
		self.conversation_language = "auto"
		self.talkative_level = "peak"
		self.session_turns: Deque[ConversationTurn] = deque(maxlen=config.memory_window)
		self.last_response_hash: Optional[int] = None
		self._last_local_offer_turn: int = -999
		self._local_offer_cooldown_turns: int = 4

	def _resolve_conversation_language(self, user_text: str) -> str:
		if self.conversation_language in {"english", "hinglish", "hindi"}:
			return self.conversation_language
		text = (user_text or "").strip()
		if not text:
			return "english"
		if re.search(r"[\u0900-\u097F]", text):
			return "hindi"
		tokens = set(tokenise(text))
		hindi_roman_markers = {
			"kya", "kaise", "kyu", "kyun", "nahi", "nahin", "haan", "ha", "achha", "accha",
			"thik", "theek", "jaldi", "abhi", "chahiye", "kar", "karo", "karta", "kr", "krna",
			"bolo", "batao", "samjha", "samjhao", "mera", "meri", "mujhe", "tum", "aap", "bhai",
			"yaar", "sahi", "galat", "thoda", "zyada", "seedha", "simple", "madad",
		}
		if len(tokens & hindi_roman_markers) >= 2:
			return "hinglish"
		return "english"

	def _extract_weather_location(self, user_text: str) -> Optional[str]:
		text = (user_text or "").strip()
		if not text:
			return None

		patterns = [
			r"\b(?:weather|forecast|temperature|rain|humidity|wind)\s+(?:in|at|for|of)\s+(.+)$",
			r"\b(?:what(?:\s+is|\s+'s)?\s+)?(?:the\s+)?(?:weather|forecast|temperature)\s+(.+)$",
		]

		candidate = ""
		for pattern in patterns:
			match = re.search(pattern, text, flags=re.I)
			if match:
				candidate = (match.group(1) or "").strip()
				if candidate:
					break

		if not candidate:
			return None

		candidate = re.sub(r"\b(?:today|now|right\s+now|currently|outside|please)\b.*$", "", candidate, flags=re.I).strip(" ,?.!")
		candidate = re.sub(r"^(?:in|at|for|of)\s+", "", candidate, flags=re.I).strip()
		if not candidate:
			return None
		if normalize_text(candidate) in {"today", "now", "outside", "here", "current"}:
			return None
		return candidate

	def _extract_traffic_location(self, user_text: str) -> Optional[str]:
		text = (user_text or "").strip()
		if not text:
			return None

		patterns = [
			r"\b(?:traffic|troffic|trafic|congestion|jam|road traffic|route traffic|travel time|eta)\s+(?:in|at|for|of|near)\s+(.+)$",
			r"\b(?:how(?:'s| is)?\s+)?(?:traffic|troffic|trafic)\s+(.+)$",
			r"^\s*(.+?)\s+(?:traffic|troffic|trafic)\s*$",
		]

		candidate = ""
		for pattern in patterns:
			match = re.search(pattern, text, flags=re.I)
			if match:
				candidate = (match.group(1) or "").strip()
				if candidate:
					break

		if not candidate:
			return None

		candidate = re.sub(r"\b(?:today|now|right\s+now|currently|please)\b.*$", "", candidate, flags=re.I).strip(" ,?.!")
		candidate = re.sub(r"^(?:in|at|for|of|near)\s+", "", candidate, flags=re.I).strip()
		if not candidate:
			return None
		if normalize_text(candidate) in {"today", "now", "here", "current", "my city", "my location"}:
			return None
		return candidate

	def _extract_route_endpoints(self, user_text: str) -> Optional[Tuple[str, str]]:
		text = (user_text or "").strip()
		if not text:
			return None

		patterns = [
			r"\bfrom\s+(.+?)\s+to\s+(.+)$",
			r"\bbetween\s+(.+?)\s+and\s+(.+)$",
			r"\broute\s+(.+?)\s+to\s+(.+)$",
			r"\b(?:traffic|troffic|trafic)\s+(.+?)\s+to\s+(.+)$",
			r"^\s*(.+?)\s+to\s+(.+?)\s+(?:traffic|troffic|trafic)\s*$",
		]

		for pattern in patterns:
			match = re.search(pattern, text, flags=re.I)
			if not match:
				continue
			origin = (match.group(1) or "").strip(" ,?.!")
			destination = (match.group(2) or "").strip(" ,?.!")
			origin = re.sub(r"\b(?:traffic|route|road|drive|driving|travel|eta|time)\b", "", origin, flags=re.I).strip(" ,?.!")
			destination = re.sub(r"\b(?:traffic|route|road|drive|driving|travel|eta|time|now|today|please)\b", "", destination, flags=re.I).strip(" ,?.!")
			if origin and destination and normalize_text(origin) != normalize_text(destination):
				return origin, destination
		return None

	def ingest(self, user_text: str, source: str = "text") -> ActionPlan:
		user_text = user_text.strip()
		if not user_text:
			return ActionPlan(intent="empty", response="Say something and I will respond.", confidence=0.0)

		signal = self.user_model.infer_style(user_text)
		signal.source = source
		heuristic_intent, heuristic_confidence, heuristic_details = self.router.route(user_text)
		intent, confidence, details = self.intent_classifier.classify(user_text, heuristic_intent, heuristic_confidence, heuristic_details)
		self.session_turns.append(ConversationTurn(role="user", content=user_text, metadata={"intent": intent}))
		self.memory.add_turn("user", user_text, metadata={"intent": intent, "source": source})
		if self.config.learning_enabled:
			self.user_model.observe_turn(user_text, signal, intent)
		plan = self._plan(user_text, intent, confidence, signal, details)
		plan.response = self.persona.style_response(plan.response, signal, self.user_model.profile_snapshot())
		plan.response = self._avoid_repetition(plan.response, plan.intent)
		plan.response = self._ensure_local_updates_offer(user_text, plan.intent, plan.response)
		self.memory.add_turn("assistant", plan.response, metadata={"intent": plan.intent, "confidence": plan.confidence, "needs_confirmation": plan.needs_confirmation})
		self.session_turns.append(ConversationTurn(role="assistant", content=plan.response, metadata={"intent": plan.intent}))
		return plan

	def _ensure_local_updates_offer(self, user_text: str, intent: str, response: str) -> str:
		if intent != "conversation":
			return response
		normalized = normalize_text(user_text)
		is_basic_chat = (
			({"hi", "hello", "hey"} & set(tokenise(normalized)))
			or any(phrase in normalized for phrase in [
				"good morning", "good evening", "good afternoon",
				"how are you", "how are u", "what's up", "whats up", "kaise ho",
			])
		)
		if not is_basic_chat:
			return response
		response_norm = normalize_text(response)
		if any(term in response_norm for term in ["news", "update", "updates", "local", "location", "city", "headlines"]):
			return response

		# Avoid repeating the local-news suggestion too often in back-to-back small talk.
		turn_index = len(self.session_turns)
		if turn_index - self._last_local_offer_turn < self._local_offer_cooldown_turns:
			return response

		# Randomized injection frequency and phrasing for natural variation.
		if random.random() > 0.45:
			return response

		is_hinglish = any(marker in normalized for marker in ["kaise ho", "kya", "tum", "main", "hai"])
		if is_hinglish:
			offers = [
				"Tum chaho to main tumhari location ke local news aur live updates bhi de sakta hoon.",
				"Agar chaho to main abhi ke local city updates nikaal doon.",
				"Chaaho to tumhare area ki headlines aur live local updates laa sakta hoon.",
			]
		else:
			offers = [
				"I can also share local city news and live updates for your current location.",
				"If you want, I can pull nearby headlines and live city updates.",
				"I can fetch local updates around your area anytime.",
			]
		self._last_local_offer_turn = turn_index
		return response.rstrip() + " " + random.choice(offers)

	def _plan(self, user_text: str, intent: str, confidence: float, signal: UserSignal, details: Dict[str, Any]) -> ActionPlan:
		normalized = normalize_text(user_text)
		if normalized in {"talkative mode on", "talkative mode peak", "talkative peak", "be more talkative"}:
			self.talkative_level = "peak"
			return ActionPlan(intent="conversation", response="Talkative mode is now at peak. I will keep casual conversations richer and more human-like.", confidence=0.95)
		if normalized in {"talkative mode normal", "talkative mode balanced", "talkative normal"}:
			self.talkative_level = "balanced"
			return ActionPlan(intent="conversation", response="Talkative mode set to balanced. I will stay natural but more compact.", confidence=0.95)
		if normalized in {"talkative mode off", "concise mode", "be concise", "less talkative"}:
			self.talkative_level = "concise"
			return ActionPlan(intent="conversation", response="Concise mode enabled. I will keep replies tighter.", confidence=0.95)
		if normalized in {"fast mode", "mode fast", "set fast mode", "enable fast mode"}:
			self.reasoning_mode = "fast"
			return ActionPlan(intent="conversation", response="Fast mode enabled. I will prioritize lower latency responses.", confidence=0.95)
		if normalized in {"deep mode", "mode deep", "set deep mode", "enable deep mode"}:
			self.reasoning_mode = "deep"
			return ActionPlan(intent="conversation", response="Deep mode enabled. I will prioritize richer and more detailed responses.", confidence=0.95)
		if normalized in {"balanced mode", "mode balanced", "set balanced mode", "enable balanced mode"}:
			self.reasoning_mode = "balanced"
			return ActionPlan(intent="conversation", response="Balanced mode enabled. I will keep speed and depth in equilibrium.", confidence=0.95)
		if normalized in {"hinglish mode", "mode hinglish", "set hinglish mode", "enable hinglish mode", "speak hinglish"}:
			self.conversation_language = "hinglish"
			return ActionPlan(intent="conversation", response="Hinglish mode on. Ab main natural Hinglish style me respond karunga.", confidence=0.95)
		if normalized in {"hindi mode", "mode hindi", "set hindi mode", "enable hindi mode", "speak hindi"}:
			self.conversation_language = "hindi"
			return ActionPlan(intent="conversation", response="Hindi mode on. Ab main Hindi me naturally respond karunga for clearer voice output.", confidence=0.95)
		if normalized in {"english mode", "mode english", "set english mode", "enable english mode", "speak english", "normal english"}:
			self.conversation_language = "english"
			return ActionPlan(intent="conversation", response="English mode on. I will respond in natural conversational English.", confidence=0.95)
		if normalized in {"auto language mode", "language auto", "mode auto language", "auto mode language", "auto english hindi", "auto hinglish"}:
			self.conversation_language = "auto"
			return ActionPlan(intent="conversation", response="Auto language mode on. I will switch between English, Hindi, and Hinglish based on your input.", confidence=0.95)
		if intent == "summary":
			target_match = re.search(r"\b(?:summarize|summary|sum up|in short|tl;dr)\b[:\-\s]*(.*)$", user_text, flags=re.I)
			target_text = (target_match.group(1).strip() if target_match else "") if target_match else ""
			if not target_text:
				recent = self.memory.get_recent_turns(8)
				target_text = "\n".join([f"{turn.role}: {turn.content}" for turn in recent])
			if not target_text:
				return ActionPlan(intent="summary", response="There is nothing available to summarize yet.", confidence=0.8)
			messages = [
				{"role": "system", "content": "Summarize the content clearly with key points and keep important facts. If the user asked short summary, keep it concise."},
				{"role": "user", "content": target_text},
			]
			summary_text = self.research.llm.complete(messages, temperature=0.2, max_tokens=420)
			if not summary_text:
				summary_text = "I could not summarize that reliably right now."
			return ActionPlan(intent="summary", response=summary_text, confidence=max(confidence, 0.7))

		if intent in {"research", "analysis"}:
			deep_markers = ["deep research", "with sources", "verify with sources", "cite sources", "web research", "detailed research"]
			requires_deep_research = self.reasoning_mode == "deep" or any(marker in normalized for marker in deep_markers)
			if requires_deep_research:
				if self.reasoning_mode == "fast":
					max_results = min(3, self.config.max_search_results)
				elif self.reasoning_mode == "deep":
					max_results = max(self.config.max_search_results, 6)
				else:
					max_results = self.config.max_search_results
				result = self.research.answer(user_text, max_results=max_results)
			else:
				result = self.research.fast_answer(user_text)
			return ActionPlan(intent=intent, response=result["answer"], confidence=max(confidence, 0.65), metadata={"sources": result.get("sources", [])})
		if intent == "memory":
			if "forget" in normalized:
				match = re.search(r"forget (.+)", normalized)
				if match:
					key = match.group(1).strip()
					self.memory.upsert_fact("user_profile", key, None, confidence=0.1, source="forget")
					return ActionPlan(intent="memory", response=f"I removed the preference for {key}.", confidence=0.8)
			return ActionPlan(intent="memory", response="I can store, recall, and update your preferences and facts over time.", confidence=max(confidence, 0.65))
		if intent == "system":
			if re.search(r"\b(show|explain|status)\s+(pipeline|model pipeline|task pipeline|routing)\b", normalized):
				models = self.research.llm.pipeline_overview()
				lines = [
					"Execution pipeline:",
					"1. Intent routing: heuristic + intent model.",
					f"2. Fast task-router model: {models.get('router')} decides the best lane.",
					f"3. Model routing: fast={models.get('fast')}, deep={models.get('deep')}, planning={models.get('planning')}, coding={models.get('coding')}, reasoning={models.get('reasoning')}, long_context={models.get('long_context')}, default={models.get('default')}.",
					"4. Fallback chain: routed lane candidates -> default candidates -> local rule-based fallback.",
				]
				return ActionPlan(intent="system", response="\n".join(lines), confidence=0.99, metadata={"models": models})

			has_gpu = bool(re.search(r"\b(gpu|graphics|vram|video memory|gpu usage)\b", normalized))
			has_ip = bool(re.search(r"\b(ip address|public ip|local ip|my ip|internet ip|network ip|ip)\b", normalized))
			if has_gpu and has_ip:
				gpu = self.system.get_gpu_report()
				ip = self.system.get_ip_report()
				combined = "\n\n".join([gpu.get("report", "GPU report unavailable."), ip.get("report", "IP report unavailable.")])
				return ActionPlan(intent="system", response=combined, confidence=0.98, metadata={"gpu": gpu, "ip": ip})

			if has_gpu:
				gpu = self.system.get_gpu_report()
				return ActionPlan(intent="system", response=gpu.get("report", "GPU report unavailable."), confidence=0.97 if gpu.get("ok") else 0.65, metadata=gpu)

			if has_ip:
				ip = self.system.get_ip_report()
				return ActionPlan(intent="system", response=ip.get("report", "IP report unavailable."), confidence=0.97 if ip.get("ok") else 0.65, metadata=ip)

			if re.search(r"\b(weather|forecast|temperature|rain|humidity|wind)\b", normalized):
				location = self._extract_weather_location(user_text)
				weather = self.system.get_weather_report(location)
				if weather.get("ok"):
					return ActionPlan(intent="system", response=weather["report"], confidence=0.98, metadata=weather)
				return ActionPlan(intent="system", response=f"I could not fetch weather right now: {weather.get('error', 'unknown error')}", confidence=0.6, metadata=weather)

			if re.search(r"\b(traffic|troffic|trafic|congestion|jam|road traffic|route traffic|travel time|eta)\b", normalized):
				route = self._extract_route_endpoints(user_text)
				if route:
					origin, destination = route
					traffic = self.system.get_route_traffic_report(origin, destination)
				else:
					city = self._extract_traffic_location(user_text)
					traffic = self.system.get_city_traffic_report(city)
				if traffic.get("report"):
					response_text = traffic["report"]
					if re.search(r"\b(frontend|webapp|dashboard|ui)\b", normalized) and traffic.get("frontend_payload"):
						response_text += "\n\nFrontend payload is available in metadata.frontend_payload for rendering traffic cards/buttons."
					return ActionPlan(intent="system", response=response_text, confidence=0.94 if traffic.get("ok") else 0.7, metadata=traffic)
				return ActionPlan(intent="system", response=f"I could not fetch traffic right now: {traffic.get('error', 'unknown error')}", confidence=0.62, metadata=traffic)

			if re.search(r"\b(news|city news|local news|updates|local updates|headlines|current events)\b", normalized):
				return ActionPlan(intent="system", response="Fetching local city news and updates for your location. Check the City Live Updates panel.", confidence=0.92, metadata={"action": "fetch_city_news"})

			if re.search(r"\b(global news|worldwide news|world news|trending|tech news|crypto news|business news)\b", normalized):
				category = "all"
				if "tech" in normalized:
					category = "tech"
				elif "crypto" in normalized:
					category = "crypto"
				elif "business" in normalized:
					category = "business"
				return ActionPlan(intent="system", response=f"Fetching global {category} news and trending updates worldwide. Check the Global News panel.", confidence=0.92, metadata={"action": "fetch_global_news", "category": category})

			web_action = self.system.execute_web_voice_command(user_text)
			if web_action is not None:
				if web_action.get("ok"):
					action = web_action.get("action", "web")
					if action == "open_url":
						url = web_action.get("url", "")
						return ActionPlan(intent="system", response=f"Opening {url}.", confidence=0.94, metadata=web_action)
					if action == "key":
						key = web_action.get("key", "")
						return ActionPlan(intent="system", response=f"Browser command sent: {key}.", confidence=0.9, metadata=web_action)
					return ActionPlan(intent="system", response="Web command executed.", confidence=0.88, metadata=web_action)
				return ActionPlan(intent="system", response=f"I recognized a browser command but could not execute it: {web_action.get('error', 'unknown error')}", confidence=0.62, metadata=web_action)

			if re.search(r"\b(model|which model|what model|backend model|llm)\b", normalized):
				models = self.research.llm.pipeline_overview()
				return ActionPlan(intent="system", response=(
					"Pipeline models: "
					f"default={models.get('default')}, "
					f"fast={models.get('fast')}, "
					f"deep={models.get('deep')}, "
					f"planning={models.get('planning')}, "
					f"coding={models.get('coding')}."
				), confidence=0.99, metadata={"models": models})
			if re.search(r"\b(year|which year|current year|what year is this)\b", normalized):
				return ActionPlan(intent="system", response="Current year is 2026.", confidence=0.99)
			if re.search(r"\b(time|what time is it|what is the time|current time)\b", normalized):
				now = datetime.now()
				return ActionPlan(intent="system", response=f"Current time is {now.strftime('%I:%M %p')}.", confidence=0.98)
			if re.search(r"\b(date|today'?s date|current date|what date is it)\b", normalized):
				now = datetime.now()
				return ActionPlan(intent="system", response=f"Today's date is {now.strftime('%A, %d %B %Y')}.", confidence=0.98)
			if re.search(r"\b(day today|what day is it|which day is today)\b", normalized):
				now = datetime.now()
				return ActionPlan(intent="system", response=f"Today is {now.strftime('%A')}.", confidence=0.98)
			if any(word in normalized for word in ["open http", "open https", "open website", "open url"]):
				url_match = re.search(r"https?://\S+", user_text)
				if url_match:
					result = self.system.open_url(url_match.group(0))
					return ActionPlan(intent="system", response="Opening the requested URL.", confidence=0.9, metadata=result)
			if any(word in normalized for word in ["open ", "launch "]):
				app_match = re.search(r"\b(?:open|launch)\s+(.+)", user_text, flags=re.I)
				if app_match:
					target = app_match.group(1).strip()
					target_clean = re.sub(r"\b(from|in|using|with)\s+(the\s+)?(brave|chrome|edge|browser)\b.*$", "", target, flags=re.I).strip()
					target_lower = normalize_text(target_clean)
					if target_lower in SystemController.WEB_SHORTCUTS:
						result = self.system.open_url(SystemController.WEB_SHORTCUTS[target_lower])
						return ActionPlan(intent="system", response=f"Opening {target_clean}.", confidence=0.9, metadata=result)
					for key, site_url in SystemController.WEB_SHORTCUTS.items():
						if key in target_lower:
							result = self.system.open_url(site_url)
							return ActionPlan(intent="system", response=f"Opening {key}.", confidence=0.88, metadata=result)
					result = self.system.open_app(target_clean or target)
					if result.get("ok"):
						return ActionPlan(intent="system", response=f"Opening {target_clean or target}.", confidence=0.88, metadata=result)
					return ActionPlan(intent="system", response=f"I could not open {target_clean or target}. {result.get('error', '')}", confidence=0.65, metadata=result)
			if "clipboard" in normalized:
				if "read" in normalized:
					clip = self.system.read_clipboard()
					return ActionPlan(intent="system", response=clip or "Clipboard is empty.", confidence=0.8)
				if "write" in normalized:
					text_to_copy = user_text.split("write", 1)[-1].strip()
					ok = self.system.write_clipboard(text_to_copy)
					return ActionPlan(intent="system", response="Clipboard updated." if ok else "Clipboard update failed.", confidence=0.75)
			if any(word in normalized for word in ["stats", "system stats", "cpu", "memory"]):
				return ActionPlan(intent="system", response=json.dumps(self.system.system_stats(), indent=2), confidence=0.8)
			if any(word in normalized for word in ["delete", "remove", "move", "rename", "close"]):
				return ActionPlan(intent="system", response="That action needs confirmation. State the exact target and I will verify it first.", confidence=0.9, needs_confirmation=True)
			return ActionPlan(intent="system", response="System tools are ready. Provide a precise command, app, path, or URL.", confidence=max(confidence, 0.7))
		if intent == "planning":
			messages = [
				{"role": "system", "content": "Create a phase-by-phase plan with priorities, risks, and next steps. Answer fully and clearly."},
				{"role": "user", "content": user_text},
			]
			plan_text = self.research.llm.complete_for_task("planning", messages, temperature=0.25, max_tokens=700, reasoning_mode=self.reasoning_mode)
			return ActionPlan(intent="planning", response=plan_text, confidence=max(confidence, 0.65))
		if intent == "conversation":
			active_language = self._resolve_conversation_language(user_text)
			basic_conversation = (
				({"hi", "hello", "hey"} & set(tokenise(normalized)))
				or any(phrase in normalized for phrase in [
					"good morning", "good evening", "good afternoon",
					"how are you", "how are u", "what's up", "whats up", "kaise ho",
				])
			)
			if basic_conversation:
				fast_reply = self.dialogue.quick_reply(user_text, signal, self.user_model.profile_snapshot(), active_language)
				if fast_reply is not None:
					return ActionPlan(
						intent="conversation",
						response=fast_reply,
						confidence=max(confidence, 0.78),
						metadata={"mode": "basic-conversation-fast-path", "reasoning_mode": self.reasoning_mode, "conversation_language": active_language, "language_setting": self.conversation_language},
					)
			if self.dialogue.should_bypass_llm(user_text, intent, self.talkative_level):
				fallback_reply = self.dialogue.quick_reply(user_text, signal, self.user_model.profile_snapshot(), active_language)
				if fallback_reply is not None:
					return ActionPlan(intent="conversation", response=fallback_reply, confidence=max(confidence, 0.76), metadata={"mode": "fast-path", "reasoning_mode": self.reasoning_mode, "conversation_language": active_language, "language_setting": self.conversation_language})
			if self.reasoning_mode == "fast":
				temperature = 0.5
				max_tokens = 260
			elif self.reasoning_mode == "deep":
				temperature = 0.36
				max_tokens = 900
			else:
				temperature = 0.42
				max_tokens = 420
			if self.talkative_level == "peak" and intent == "conversation":
				max_tokens = max(max_tokens, 780)
				temperature = max(temperature, 0.5)
			elif self.talkative_level == "concise":
				max_tokens = min(max_tokens, 260)
			messages = [
				{"role": "system", "content": f"You are {APP_NAME}, an Iron-Man-inspired assistant in a live back-and-forth conversation. Sound human, responsive, and adaptive. Avoid repeating the same opener. Do not say 'Of course' unless it truly fits. Match the user's tone. If the user is negotiating or tightening requirements, respond like a real collaborator: direct, concise, and specific. Ask one clarifying question only when needed. Answer fully for substantive prompts, but keep casual chat natural. In simple greetings/status chat (hi/hello/how are you), include a short offer that you can also provide local city news and live updates for the user's location."},
				{"role": "system", "content": f"Conversation language mode: {active_language} (setting: {self.conversation_language}). If hindi, respond in natural, simple Hindi in Devanagari script. If hinglish, respond in natural Hinglish and prefer Devanagari for Hindi words when voice clarity matters. If english, respond in natural conversational English."},
				{"role": "system", "content": f"Talkative level: {self.talkative_level}. If peak, be warmly conversational and fuller in casual replies. If concise, keep responses brief. If balanced, keep moderate detail."},
				{"role": "system", "content": f"Reasoning mode: {self.reasoning_mode}. In fast mode, optimize for quick turnaround. In deep mode, optimize for depth and context."},
				{"role": "system", "content": f"User profile: {json.dumps(self.user_model.profile_snapshot(), ensure_ascii=False)}"},
				{"role": "user", "content": user_text},
			]
			response = self.research.llm.complete_for_task("conversation", messages, temperature=temperature, max_tokens=max_tokens, reasoning_mode=self.reasoning_mode)
			if not response:
				response = self.dialogue.quick_reply(user_text, signal, self.user_model.profile_snapshot(), active_language) or "I’m here. Give me the next instruction."
			return ActionPlan(intent="conversation", response=response, confidence=max(confidence, 0.55), metadata={"reasoning_mode": self.reasoning_mode, "conversation_language": active_language, "language_setting": self.conversation_language})
		if intent == "command":
			return ActionPlan(intent="command", response="I am ready. I can research, remember, control tools, summarize, or speak.", confidence=confidence)
		return ActionPlan(intent="conversation", response="I am here with you. Tell me what you want, and I will handle it directly.", confidence=confidence)

	def _memory_plan(self, user_text: str, confidence: float) -> ActionPlan:
		return ActionPlan(intent="memory", response="I can store preferences, facts, feedback, and skill results to improve future responses.", confidence=confidence)

	def _avoid_repetition(self, response: str, intent: str = "conversation") -> str:
		if intent == "system":
			return response
		if len(response.strip()) <= 90:
			return response
		if response.startswith(("Current time is", "Today's date is", "Today is", "Opening ")):
			return response
		response_hash = hash(normalize_text(response))
		if response_hash == self.last_response_hash:
			response = response + " I can reframe that if you want a different angle."
		self.last_response_hash = response_hash
		return response

	def feedback(self, rating: float, note: str = "") -> None:
		if self.config.learning_enabled:
			self.user_model.learn_from_feedback(rating, note)
		else:
			self.memory.log_feedback(None, rating, note)

	def summary(self) -> str:
		preferences = self.user_model.profile_snapshot()
		recent = self.memory.get_recent_turns(5)
		return "\n".join([
			f"{APP_NAME} state summary:",
			f"- Reasoning mode: {self.reasoning_mode}",
			f"- Talkative level: {self.talkative_level}",
			f"- Conversation language setting: {self.conversation_language}",
			f"- Profile: {json.dumps(preferences, ensure_ascii=False)}",
			f"- Recent turns: {len(recent)}",
		])


class AdvancedBrain(Brain):
	"""Expanded brain with voice helpers."""

	def __init__(self, config: JarvisConfig, memory: MemoryStore):
		super().__init__(config, memory)
		self.perception = None
		self.voice = VoicePipeline()
		self.notifications = None
		self.resources = None
		self.self_learning_summary: List[str] = []

	def record_voice_text(self, transcript: str) -> ActionPlan:
		return self.ingest(transcript, source="voice")

	def daily_briefing(self) -> str:
		return self.research.daily_briefing()


class JarvisApp:
	"""User-facing application wrapper."""

	def __init__(self, config: Optional[JarvisConfig] = None):
		self.config = config or JarvisConfig.load()
		self.memory = MemoryStore(self.config.db_path)
		self.brain = Brain(self.config, self.memory)
		self._running = False
		self._event_queue: "queue.Queue[str]" = queue.Queue()

	def process_text(self, text: str) -> ActionPlan:
		return self.brain.ingest(text)

	def teach_preference(self, key: str, value: Any, confidence: float = 0.7) -> None:
		self.memory.set_preference(key, value, confidence)

	def store_fact(self, category: str, key: str, value: Any, confidence: float = 0.6) -> None:
		self.memory.upsert_fact(category, key, value, confidence)

	def get_context(self) -> Dict[str, Any]:
		return {
			"profile": self.brain.user_model.profile_snapshot(),
			"preferences": self.memory.get_preferences(),
			"facts": self.memory.get_facts(),
			"recent_turns": [dataclasses.asdict(turn) for turn in self.memory.get_recent_turns(self.config.memory_window)],
		}

	def run_cli(self) -> None:
		self._running = True
		print(f"{self.config.persona_name} is ready. Type 'exit' to quit.")
		while self._running:
			try:
				user_input = input("You: ").strip()
			except (EOFError, KeyboardInterrupt):
				print("\nExiting.")
				break
			if not user_input:
				continue
			command = user_input.lower()
			if command in {"exit", "quit", "bye"}:
				print("Jarvis: Until next time.")
				break
			plan = self.process_text(user_input)
			response_text = (plan.response or "").strip() or "I did not generate a response. Please repeat your request."
			print(f"Jarvis: {response_text}")
			if isinstance(self, AdvancedJarvisApp):
				self.hud.set_response(response_text)
				self.voice.speak(response_text)

	def stop(self) -> None:
		self._running = False


class AdvancedJarvisApp(JarvisApp):
	"""Full-featured Jarvis application wrapper."""

	def __init__(self, config: Optional[JarvisConfig] = None):
		super().__init__(config)
		self.brain = AdvancedBrain(self.config, self.memory)
		self.hud = HudOverlay()
		self.hud.set_command_callback(self._enqueue_text_command)
		self.voice = self.brain.voice
		self.notifications = None
		self.dependencies = None
		self.voice_hotkey = DEFAULT_VOICE_HOTKEY
		self._hotkey_event = threading.Event()
		self._hotkey_listener_started = False
		self._text_input_queue: "queue.Queue[str]" = queue.Queue()
		self._text_listener_started = False
		self._gesture_monitor_thread = None
		self._emotion_monitor_thread = None
		self._gesture_enabled = False
		self._emotion_enabled = False
		self._latest_gesture: Tuple[str, float] = ("idle", 0.0)
		self._latest_emotion: Tuple[str, float] = ("neutral", 0.5)
		self._latest_visual_hint = ""
		self._last_visual_hint_ts = 0.0

	def _enqueue_text_command(self, command_text: str) -> None:
		value = (command_text or "").strip()
		if value:
			self._text_input_queue.put(value)

	def _learning_telemetry_text(self, plan: Optional[ActionPlan] = None) -> str:
		profile = self.brain.user_model.profile_snapshot()
		learning_state = "ON" if self.config.learning_enabled else "OFF"
		depth = float(profile.get("technical_depth", 0.5))
		stress = float(profile.get("stress_sensitivity", 0.5))
		formality = float(profile.get("formality", 0.5))
		intent = plan.intent if plan is not None else "idle"
		confidence = f"{plan.confidence:.2f}" if plan is not None else "-"
		return "\n".join([
			f"Learning: {learning_state}",
			f"Last intent: {intent} (conf {confidence})",
			f"Depth {depth:.2f} | StressSense {stress:.2f}",
			f"Formality {formality:.2f} | Mode {self.brain.reasoning_mode}",
		])

	def _apply_visual_state(self) -> None:
		"""Apply camera-derived visual state on the main thread."""
		gesture_name, gesture_conf = self._latest_gesture
		self.hud.set_gesture(gesture_name, gesture_conf)
		emotion_name, emotion_conf = self._latest_emotion
		self.hud.set_emotion(emotion_name, emotion_conf)

		if self._latest_visual_hint and (time.time() - self._last_visual_hint_ts) > 1.2:
			self.hud.set_response(self._latest_visual_hint)
			self._last_visual_hint_ts = time.time()

	def _start_visual_features(self, require_face_auth: bool = False) -> bool:
		"""Placeholder - visual features removed."""
		return True

	def _stop_visual_features(self) -> None:
		"""Placeholder - visual features removed."""
		pass

	def start_voice_mode(self, enable_text_input: bool = True) -> None:
		self._running = True
		self._start_hotkey_listener()
		text_input_enabled = enable_text_input
		if text_input_enabled:
			self._start_text_listener()
		
		self.hud.set_mode("active", "JARVIS ACTIVE // Waiting for wake phrase")
		self.hud.set_learning_info(self._learning_telemetry_text())
		print(f"{self.config.persona_name}: Voice mode active.")
		print(f"{self.config.persona_name}: Say 'Hey Jarvis' or press {self.voice_hotkey} to trigger command capture.")
		
		conversation_window_seconds = 60.0
		conversation_active_until = 0.0
		fallback_announced = False
		
		while self._running:
			if not self.voice.microphone_available:
				if not text_input_enabled:
					self._start_text_listener()
					text_input_enabled = True
				if not fallback_announced:
					reason = self.voice.microphone_error or "microphone unavailable"
					self.hud.set_mode("active", "MIC OFFLINE // TEXT FALLBACK ENABLED")
					self.hud.set_response(f"Audio input unavailable: {reason}. Type commands in terminal.")
					print(f"{self.config.persona_name}: Microphone unavailable ({reason}). Text fallback enabled.")
					fallback_announced = True

			text_command = self._poll_text_command()
			if text_command is not None:
				normalized_text = normalize_text(text_command)
				if normalized_text in {"retry mic", "mic retry", "reconnect mic", "retry microphone"}:
					self.voice.reset_microphone_state()
					fallback_announced = False
					self.hud.set_mode("active", "MIC RETRY REQUESTED // Waiting for wake phrase")
					self.hud.set_response("Microphone retry requested. Try saying wake phrase now.")
					continue
				if normalized_text in {"exit", "quit", "bye", "exit jarvis", "stop jarvis", "quit jarvis", "shutdown jarvis"}:
					self._running = False
					self.hud.set_mode("idle", "SESSION STOPPED")
					self.hud.set_response("Session ended.")
					break
				self._handle_text_command(text_command, speak_response=True)
				conversation_active_until = time.time() + conversation_window_seconds
				continue

			manual_trigger = False
			now = time.time()
			if self._hotkey_event.is_set():
				manual_trigger = True
				self._hotkey_event.clear()
				self.hud.set_mode("listening", "HOTKEY TRIGGERED // Listening")
				transcript = self.voice.listen_once(timeout=6, phrase_time_limit=18, retries=1)
			elif now < conversation_active_until:
				remaining = max(1, int(conversation_active_until - now))
				self.hud.set_mode("listening", f"CONVERSATION ACTIVE // {remaining}s")
				transcript = self.voice.listen_once(timeout=6, phrase_time_limit=15, retries=1)
			else:
				self.hud.set_mode("active", "JARVIS ACTIVE // Waiting for wake phrase")
				transcript = self.voice.listen_once(timeout=4, phrase_time_limit=12, retries=1)
			if not transcript:
				error_lower = self.voice.last_error.lower() if self.voice.last_error else ""
				if self.voice.last_error and "timed out" not in error_lower and "speech not understood" not in error_lower:
					self.hud.set_response(f"Audio input issue: {self.voice.last_error}")
				continue
			normalized = normalize_text(transcript)
			if normalized in {"exit jarvis", "stop jarvis", "quit jarvis", "shutdown jarvis"}:
				self._running = False
				self.hud.set_mode("idle", "VOICE MODE STOPPED")
				self.hud.set_response("Session ended.")
				self.voice.speak("Voice mode stopped.")
				break
			if manual_trigger:
				self._handle_voice_command(transcript)
				conversation_active_until = time.time() + conversation_window_seconds
				continue
			remainder = self.voice.extract_command_after_wake(transcript)
			if remainder is not None:
				self.hud.set_transcript(transcript)
				if remainder:
					self.hud.set_mode("processing", "WAKE PHRASE DETECTED // Processing inline command")
					self._handle_voice_command(remainder)
				else:
					self.hud.set_mode("listening", "WAKE PHRASE DETECTED // Listening")
					self.voice.speak("Listening.")
					command = self.voice.listen_once(timeout=6, phrase_time_limit=18, retries=2, retry_delay=0.2)
					if not command:
						self.hud.set_mode("active", "I DID NOT CATCH THAT")
						self.hud.set_response("No command captured. Please repeat your request.")
						self.voice.speak("I did not catch that.")
						continue
					self._handle_voice_command(command)
				conversation_active_until = time.time() + conversation_window_seconds
				continue
			if now < conversation_active_until:
				self._handle_voice_command(transcript)
				conversation_active_until = time.time() + conversation_window_seconds
		self.hud.close()

	def _start_gesture_monitoring(self) -> None:
		"""Placeholder - gesture monitoring removed."""
		pass

	def _start_emotion_monitoring(self) -> None:
		"""Placeholder - emotion detection removed."""
		pass

	def start_text_mode(self) -> None:
		self._running = True
		self._start_text_listener()
		self.hud.set_mode("active", "TEXT MODE ACTIVE")
		self.hud.set_learning_info(self._learning_telemetry_text())
		self.hud.set_response("Text mode is active. Type your commands below. Visual features are also running.")
		print(f"{self.config.persona_name}: Text mode active. Type 'exit' to quit.")
		if not self._start_visual_features(require_face_auth=True):
			self._running = False
			self.hud.close()
			return
		while self._running:
			user_input = self._poll_text_command()
			if user_input is None:
				time.sleep(0.03)
				continue
			normalized = normalize_text(user_input)
			if normalized in {"exit", "quit", "bye", "exit jarvis", "stop jarvis", "quit jarvis", "shutdown jarvis"}:
				break
			self._handle_text_command(user_input, speak_response=True)
		self._running = False
		self._stop_visual_features()
		self.hud.close()

	def _handle_voice_command(self, command_text: str) -> None:
		self.hud.set_transcript(command_text)
		self.hud.set_mode("processing", "PROCESSING COMMAND")
		plan = self.brain.record_voice_text(command_text)
		self.hud.set_learning_info(self._learning_telemetry_text(plan))
		response_text = (plan.response or "").strip() or "I did not generate a response. Please repeat your request."
		print(f"Jarvis: {response_text}")
		self.hud.set_response(response_text)
		self.hud.set_mode("speaking", "SPEAKING RESPONSE")
		spoken_fully = self.voice.speak(response_text)
		if not spoken_fully and self._hotkey_event.is_set():
			self.hud.set_mode("listening", "INTERRUPTED // Listening for next command")
		else:
			self.hud.set_mode("active", "JARVIS ACTIVE // Waiting for wake phrase")

	def _handle_text_command(self, command_text: str, speak_response: bool = False) -> None:
		self.hud.set_transcript(f"[TEXT] {command_text}")
		self.hud.set_mode("processing", "PROCESSING TEXT COMMAND")
		plan = self.brain.ingest(command_text, source="text")
		self.hud.set_learning_info(self._learning_telemetry_text(plan))
		response_text = (plan.response or "").strip() or "I did not generate a response. Please repeat your request."
		print(f"Jarvis: {response_text}")
		self.hud.set_response(response_text)
		if speak_response:
			self.hud.set_mode("speaking", "SPEAKING RESPONSE")
			self.voice.speak(response_text)
		self.hud.set_mode("active", "JARVIS ACTIVE // Waiting for input")

	def _poll_text_command(self) -> Optional[str]:
		try:
			value = self._text_input_queue.get_nowait().strip()
			return value if value else None
		except queue.Empty:
			return None

	def _start_text_listener(self) -> None:
		if self._text_listener_started:
			return
		self._text_listener_started = True

		def worker() -> None:
			while self._running:
				try:
					line = input("You (text): ").strip()
				except (EOFError, KeyboardInterrupt):
					break
				if not self._running:
					break
				if line:
					self._text_input_queue.put(line)

		threading.Thread(target=worker, daemon=True).start()

	def _start_hotkey_listener(self) -> None:
		if self._hotkey_listener_started:
			return
		self._hotkey_listener_started = True

		def worker() -> None:
			try:
				import keyboard  # type: ignore
			except Exception:
				print(f"{self.config.persona_name}: Hotkey module unavailable. Use wake phrase instead.")
				return
			while self._running:
				try:
					keyboard.wait(self.voice_hotkey)
					if self._running:
						self._hotkey_event.set()
				except Exception:
					print(f"{self.config.persona_name}: Hotkey listener stopped unexpectedly.")
					break

		threading.Thread(target=worker, daemon=True).start()

	def open_url(self, url: str) -> Dict[str, Any]:
		return self.brain.system.open_url(url)

	def open_app(self, target: str) -> Dict[str, Any]:
		return self.brain.system.open_app(target)

	def speak(self, text: str) -> None:
		self.voice.speak(text)


def build_default_jarvis() -> JarvisApp:
	config = JarvisConfig.load()
	config.save()
	return AdvancedJarvisApp(config)


def main() -> None:
	parser = argparse.ArgumentParser(description="Jarvis assistant launcher")
	parser.add_argument("--mode", choices=["hybrid", "voice", "text"], default="voice", help="Input mode")
	args = parser.parse_args()
	app = build_default_jarvis()
	if isinstance(app, AdvancedJarvisApp):
		if args.mode == "text":
			app.start_text_mode()
		elif args.mode == "voice":
			app.start_voice_mode(enable_text_input=False)
		else:
			app.start_voice_mode(enable_text_input=True)
	else:
		app.run_cli()


if __name__ == "__main__":
	main()
