import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


FAQ_PATH = Path(os.getenv("FAQ_PATH", "/app/faq.txt"))
UNKNOWN_ANSWER = "Не знаю. Попробуйте спросить про время, команду, трек, сдачу или призы."
EXIT_COMMANDS = {"exit", "quit", "выход", "выйти"}


@dataclass(frozen=True)
class FaqItem:
    question: str
    answer: str


def load_faq(path: Path) -> list[FaqItem]:
    """Load blocks in the `Вопрос: ...` / `Ответ: ...` format."""
    if not path.exists():
        raise FileNotFoundError(f"Файл FAQ не найден: {path}")

    items: list[FaqItem] = []
    question: str | None = None
    answer_lines: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines() + ["Вопрос:"]:
        line = raw_line.strip()
        if line.startswith("Вопрос:"):
            if question and answer_lines:
                items.append(FaqItem(question, " ".join(answer_lines)))
            question = line.removeprefix("Вопрос:").strip() or None
            answer_lines = []
        elif line.startswith("Ответ:"):
            answer_lines.append(line.removeprefix("Ответ:").strip())
        elif line and question and answer_lines:
            answer_lines.append(line)

    if not items:
        raise ValueError("В faq.txt не найдено ни одной пары вопрос–ответ")
    return items


def normalize(text: str) -> set[str]:
    words = re.findall(r"[a-zа-яё0-9]+", text.lower())
    stop_words = {
        "а", "в", "вы", "где", "для", "и", "как", "какая", "какие", "какой",
        "когда", "ли", "на", "о", "по", "про", "с", "что", "это", "я",
    }
    return {
        word for word in words
        if len(word) > 2 and word not in stop_words and not word.startswith("репет")
    }


def local_match(query: str, items: list[FaqItem]) -> FaqItem | None:
    """Find the best FAQ using token overlap and common Russian word stems."""
    query_words = normalize(query)
    if not query_words:
        return None

    topic_groups = (
        ("врем", "начал", "старт"),
        ("команд", "участник", "человек", "люд"),
        ("трек", "направлен"),
        ("сда", "отправ", "репозитор", "дедлайн"),
        ("приз", "наград", "подар"),
    )
    best_item: FaqItem | None = None
    best_score = 0

    for item in items:
        faq_words = normalize(item.question)
        score = len(query_words & faq_words) * 3
        haystack = " ".join(faq_words)
        score += sum(
            2 for group in topic_groups
            if any(word.startswith(stem) for word in query_words for stem in group)
            and any(stem in haystack for stem in group)
        )
        if score > best_score:
            best_item, best_score = item, score

    return best_item if best_score >= 2 else None


def ai_match(query: str, items: list[FaqItem]) -> FaqItem | None:
    """Ask an OpenAI-compatible model to select an FAQ index, never to invent an answer."""
    api_key = os.getenv("AI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("AI_BASE_URL", "https://api.openai.com/v1"),
        timeout=float(os.getenv("AI_TIMEOUT", "10")),
    )
    questions = "\n".join(f"{index}: {item.question}" for index, item in enumerate(items))
    response = client.chat.completions.create(
        model=os.getenv("AI_MODEL", "gpt-4o-mini"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Выбери вопрос FAQ, совпадающий по смыслу с запросом пользователя. "
                    "Верни только JSON {\"index\": N}, где N — номер вопроса, "
                    "или -1, если запрос не относится ни к одному вопросу."
                ),
            },
            {"role": "user", "content": f"FAQ:\n{questions}\n\nЗапрос: {query}"},
        ],
    )
    index = int(json.loads(response.choices[0].message.content or "{}").get("index", -1))
    return items[index] if 0 <= index < len(items) else None


def find_answer(query: str, items: list[FaqItem]) -> str:
    local_result = local_match(query, items)
    if local_result:
        return local_result.answer

    try:
        ai_result = ai_match(query, items)
        return ai_result.answer if ai_result else UNKNOWN_ANSWER
    except Exception as error:
        if os.getenv("DEBUG", "false").lower() == "true":
            print(f"[AI недоступен: {error}]")
        return UNKNOWN_ANSWER


def main() -> None:
    items = load_faq(FAQ_PATH)
    print(f"FAQ-бот готов. Загружено вопросов: {len(items)}.")
    print("Спросите про репетицию или напишите «выход».\n")

    while True:
        try:
            query = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо встречи!")
            break
        if query.lower() in EXIT_COMMANDS:
            print("До встречи!")
            break
        if not query:
            continue
        print(f"Бот: {find_answer(query, items)}\n")


if __name__ == "__main__":
    main()
