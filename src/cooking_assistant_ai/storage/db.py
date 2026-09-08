"""SQLite persistence for recipes, inventory and session snapshots.

The spec (1.5) kept session state out of the database so it could never be a second source of
truth. It still is not one: the snapshots here are write-only while the server runs, read only
at startup, and the live Session in memory always wins. What they buy is a cook surviving a
crash or a power cut instead of losing their plan and timers.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from cooking_assistant_ai.model.types import Recipe
from cooking_assistant_ai.storage.seed import SEED_INVENTORY, SEED_RECIPES

log = logging.getLogger(__name__)


class Store:
    def __init__(self, path: str = "cooking.db", seed: bool = True):
        self.path = path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        if seed:
            self.seed_if_empty()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS recipes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS inventory (
                    name TEXT PRIMARY KEY,
                    amount REAL NOT NULL,
                    unit TEXT
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    body TEXT NOT NULL
                );
                """
            )
            self.conn.commit()

    # -- settings (household preferences that outlive a session) -------------

    def get_setting(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                              (key, value))
            self.conn.commit()

    @property
    def diet(self) -> str:
        return self.get_setting("diet", "none")

    @property
    def appliances(self) -> List[str]:
        from cooking_assistant_ai.core.appliances import owned

        return owned(self)

    def set_appliances(self, names: List[str], burners: Optional[int] = None) -> List[str]:
        from cooking_assistant_ai.core.appliances import CATALOGUE

        unknown = [n for n in names if n not in CATALOGUE]
        if unknown:
            raise ValueError(f"unknown appliance(s): {', '.join(unknown)}. "
                             f"Known: {', '.join(CATALOGUE)}")
        self.set_setting("appliances", ",".join(n for n in CATALOGUE if n in set(names)))
        if burners is not None:
            self.set_setting("burners", str(max(1, min(8, int(burners)))))
        return self.appliances

    def set_diet(self, diet: str) -> str:
        from cooking_assistant_ai.core.diet import DIETS

        d = (diet or "none").strip().lower()
        if d not in DIETS:
            raise ValueError(f"diet must be one of {', '.join(DIETS)}, got '{diet}'")
        self.set_setting("diet", d)
        return d

    def seed_if_empty(self) -> None:
        if not self.list_recipes():
            for raw in SEED_RECIPES:
                self.put_recipe(Recipe.from_dict(raw))
        if not self.inventory():
            for item in SEED_INVENTORY:
                self.set_stock(item["name"], float(item["amount"]), item["unit"])

    # -- recipes ------------------------------------------------------------

    def list_recipes(self) -> List[Recipe]:
        with self._lock:
            rows = self.conn.execute("SELECT body FROM recipes ORDER BY id").fetchall()
        return [Recipe.from_dict(json.loads(r["body"])) for r in rows]

    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        with self._lock:
            row = self.conn.execute("SELECT body FROM recipes WHERE id = ?", (recipe_id,)).fetchone()
        return Recipe.from_dict(json.loads(row["body"])) if row else None

    def find_recipe(self, ref: str) -> Optional[Recipe]:
        """By id, exact title, then title substring (case-insensitive)."""
        r = self.get_recipe(ref)
        if r:
            return r
        low = ref.strip().lower()
        recipes = self.list_recipes()
        for r in recipes:
            if r.title.lower() == low:
                return r
        for r in recipes:
            if low and low in r.title.lower():
                return r
        return None

    def put_recipe(self, recipe: Recipe) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO recipes (id, title, body) VALUES (?, ?, ?)",
                (recipe.id, recipe.title, json.dumps(recipe.to_dict())),
            )
            self.conn.commit()

    def delete_recipe(self, recipe_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,))
            self.conn.commit()
        return cur.rowcount > 0

    # -- cooking sessions (snapshots, so a crash or power cut is recoverable) ---

    def save_session(self, session_id: str, body: Dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO sessions (id, updated_at, body) VALUES (?, ?, ?)",
                (session_id, datetime.now().isoformat(), json.dumps(body)),
            )
            self.conn.commit()

    def load_sessions(self, max_age_s: Optional[float] = None) -> List[Tuple[str, datetime, Dict[str, Any]]]:
        """Snapshots newest first, as (id, saved_at, body). Corrupt rows are skipped, not raised:
        a bad snapshot must never stop the server from starting."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, updated_at, body FROM sessions ORDER BY updated_at DESC").fetchall()
        out: List[Tuple[str, datetime, Dict[str, Any]]] = []
        now = datetime.now()
        for row in rows:
            try:
                saved = datetime.fromisoformat(row["updated_at"])
                if max_age_s is not None and (now - saved).total_seconds() > max_age_s:
                    continue
                out.append((row["id"], saved, json.loads(row["body"])))
            except (ValueError, TypeError, json.JSONDecodeError):
                log.warning("skipping unreadable session snapshot %s", row["id"])
        return out

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self.conn.commit()
        return cur.rowcount > 0

    def prune_sessions(self, max_age_s: float) -> int:
        """Drop snapshots too old to be worth resuming."""
        cutoff = (datetime.now() - timedelta(seconds=max_age_s)).isoformat()
        with self._lock:
            cur = self.conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
            self.conn.commit()
        return cur.rowcount

    def next_recipe_id(self) -> str:
        ids = [r.id for r in self.list_recipes()]
        n = 1
        while f"r{n:03d}" in ids:
            n += 1
        return f"r{n:03d}"

    # -- inventory ----------------------------------------------------------

    def inventory(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute("SELECT name, amount, unit FROM inventory ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def get_stock(self, name: str) -> Optional[Dict[str, Any]]:
        low = name.strip().lower()
        with self._lock:
            row = self.conn.execute("SELECT name, amount, unit FROM inventory WHERE lower(name) = ?", (low,)).fetchone()
            if row is None:
                row = self.conn.execute(
                    "SELECT name, amount, unit FROM inventory WHERE instr(lower(name), ?) > 0 OR instr(?, lower(name)) > 0 ORDER BY length(name) LIMIT 1",
                    (low, low),
                ).fetchone()
        return dict(row) if row else None

    def set_stock(self, name: str, amount: float, unit: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO inventory (name, amount, unit) VALUES (?, ?, ?)",
                (name.strip().lower(), float(amount), unit),
            )
            self.conn.commit()
        return {"name": name.strip().lower(), "amount": float(amount), "unit": unit}

    def remove_stock(self, name: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM inventory WHERE lower(name) = ?", (name.strip().lower(),))
            self.conn.commit()
        return cur.rowcount > 0

    def deduct(self, name: str, amount: float, unit: Optional[str]) -> Dict[str, Any]:
        """Subtract from stock. Raises ValueError with a specific reason on mismatch."""
        item = self.get_stock(name)
        if item is None:
            raise ValueError(f"{name} is not in the inventory")
        if unit and item["unit"] and unit.strip().lower() != str(item["unit"]).lower():
            raise ValueError(f"{item['name']} is tracked in {item['unit']}, not {unit}")
        new_amount = max(0.0, float(item["amount"]) - float(amount))
        return self.set_stock(item["name"], new_amount, item["unit"])
