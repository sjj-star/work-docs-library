import json
import logging
from typing import Optional

from .db import KnowledgeDB
from .models import Chapter

logger = logging.getLogger(__name__)


class ChapterEditor:
    def __init__(self, db: Optional[KnowledgeDB] = None) -> None:
        self.db = db or KnowledgeDB()

    def interactive_edit(self, doc_id: str) -> None:
        doc = self.db.get_document(doc_id)
        if not doc:
            logger.error("Document not found | doc_id=%s", doc_id)
            return

        print(f"\nEditing chapters for: {doc.title}")
        print("Current chapters:")
        chapters = self.db.get_chapters(doc_id)
        for i, ch in enumerate(chapters, 1):
            print(f"  {i}. [L{ch.level}] {ch.title}  P{ch.start_page}-{ch.end_page}")

        print("\nCommands: add | edit | del | reorder | save | quit")
        working = list(chapters)

        while True:
            try:
                cmd = input("\n> ").strip().lower()
            except EOFError:
                break
            if cmd == "quit":
                break
            if cmd == "save":
                self.db.set_chapters_override(doc_id, json.dumps([c.to_dict() for c in working], ensure_ascii=False))
                print("Saved.")
                break
            if cmd == "add":
                title = input("Title: ").strip()
                try:
                    sp = int(input("Start page: "))
                    ep = int(input("End page: "))
                    lv = int(input("Level (default 1): ") or "1")
                except ValueError:
                    logger.error("Error: page numbers and level must be integers")
                    continue
                working.append(Chapter(title=title, start_page=sp, end_page=ep, level=lv))
                print("Added.")
            elif cmd == "del":
                try:
                    idx = int(input("Index to delete: ")) - 1
                except ValueError:
                    logger.error("Error: index must be an integer")
                    continue
                if 0 <= idx < len(working):
                    working.pop(idx)
                    print("Deleted.")
            elif cmd == "edit":
                try:
                    idx = int(input("Index to edit: ")) - 1
                except ValueError:
                    logger.error("Error: index must be an integer")
                    continue
                if 0 <= idx < len(working):
                    ch = working[idx]
                    ch.title = input(f"Title [{ch.title}]: ").strip() or ch.title
                    try:
                        ch.start_page = int(input(f"Start [{ch.start_page}]: ") or ch.start_page)
                        ch.end_page = int(input(f"End [{ch.end_page}]: ") or ch.end_page)
                        ch.level = int(input(f"Level [{ch.level}]: ") or ch.level)
                    except ValueError:
                        logger.error("Error: page numbers and level must be integers")
                        continue
                    print("Updated.")
            elif cmd == "reorder":
                order = input("New order as comma-separated indices: ").strip()
                try:
                    indices = [int(x.strip()) - 1 for x in order.split(",")]
                    working = [working[i] for i in indices]
                    print("Reordered.")
                except Exception as e:
                    logger.error("Error: %s", e)
            else:
                logger.warning("Unknown command: %s", cmd)
