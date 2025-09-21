import os
import argparse
import asyncio
from functools import partial
from collections import defaultdict, deque
from datetime import datetime
import json

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from rag_core import generate_answer

load_dotenv()

DIALOG_MEMORY = defaultdict(lambda: deque(maxlen=10))

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
REQ_LOG = os.path.join(LOG_DIR, "bot_requests.jsonl")

def _log_request(record: dict):
    with open(REQ_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _append_memory(chat_id: int, role: str, content: str):
    if content:
        DIALOG_MEMORY[chat_id].append({"role": role, "content": content})

def _get_recent_history(chat_id: int, last_n: int = 5):
    return list(DIALOG_MEMORY[chat_id])[-last_n:]


DEBOUNCE_SECONDS = 2.5       
FORCE_FLUSH_LEN = 220          
MAX_BUFFER_ITEMS = 8           

BUFFER = defaultdict(list)              
PENDING_TASKS = {}                       

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello / Bonjour! I am the Ecobank RAG assistant.\n"
        "I remember short context (last 5 messages) and aggregate rapid messages into a single reply.\n"
        "Ask about account opening, uploads, delays, or support.\n"
        "Commands: /help, /reset, /history"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Examples:\n"
        "- I want to open an account in Nigeria (the bot will ask follow-ups)\n"
        "- My selfie has no face — what should I do?\n"
        "- How to track my application status?\n\n"
        "Admin: /reset clears memory; /history shows recent context."
    )

async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    DIALOG_MEMORY[chat_id].clear()
    BUFFER[chat_id].clear()
    task = PENDING_TASKS.pop(chat_id, None)
    if task and not task.done():
        task.cancel()
    await update.message.reply_text("Conversation memory and input buffer have been cleared for this chat.")

async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    hist = _get_recent_history(chat_id, last_n=5)
    if not hist:
        await update.message.reply_text("History is empty.")
        return
    lines = []
    for m in hist:
        who = "You" if m["role"] == "user" else "Bot"
        lines.append(f"{who}: {m['content']}")
    await update.message.reply_text("Recent history (last 5):\n" + "\n".join(lines))

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, index_dir: str):
    """Агрегируем входящие реплики и отвечаем одним сообщением после паузы."""
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    if not text:
        return

    BUFFER[chat_id].append(text)

    must_flush_now = (
        any(len(t) >= FORCE_FLUSH_LEN for t in BUFFER[chat_id]) or
        ("?" in text) or
        (len(BUFFER[chat_id]) >= MAX_BUFFER_ITEMS)
    )

    old_task = PENDING_TASKS.get(chat_id)
    if old_task and not old_task.done():
        old_task.cancel()

    delay = 0.2 if must_flush_now else DEBOUNCE_SECONDS
    PENDING_TASKS[chat_id] = context.application.create_task(
        flush_buffer_after_delay(chat_id, context, index_dir, delay)
    )

async def flush_buffer_after_delay(chat_id: int, context: ContextTypes.DEFAULT_TYPE, index_dir: str, delay: float):
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return
    await process_buffer(chat_id, context, index_dir)

async def process_buffer(chat_id: int, context: ContextTypes.DEFAULT_TYPE, index_dir: str):
    """Склеиваем буфер, добавляем в память как одно сообщение и запускаем RAG один раз."""
    parts = BUFFER[chat_id]
    if not parts:
        return
    BUFFER[chat_id] = [] 

    combined = "\n".join(parts).strip()

    _append_memory(chat_id, "user", combined)
    chat_history = _get_recent_history(chat_id, last_n=5)

    try:
        answer, hits = generate_answer(combined, index_dir=index_dir, k=5, chat_history=chat_history)
        footer = "\n\nSources:\n" + "\n".join(
            [f"{h['rank']}. {h.get('title','')} ({h.get('doc_path','')})" for h in hits]
        )
        output = answer + footer

        _append_memory(chat_id, "assistant", answer)


        rec = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "chat_id": chat_id,
            "query": combined,
            "history_len": len(chat_history),
            "answer_len": len(answer or ""),
            "buffer_items": len(parts),
            "sources": [
                {k: h.get(k) for k in ("rank","title","doc_path","chunk_idx","score")}
                for h in hits
            ],
        }
        _log_request(rec)

        await context.bot.send_message(chat_id=chat_id, text=output)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="index_out", help="Path with faiss.index + meta.jsonl")
    args = ap.parse_args()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("history", history_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_text, index_dir=args.index_dir)))
    app.run_polling()

if __name__ == "__main__":
    main()
