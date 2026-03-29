# To-Do List with ML features
# Made this for our AIML project, added priority prediction and completion chance
# using naive bayes and logistic regression

# need to install these if not already:
# pip install scikit-learn colorama numpy

from datetime import datetime
from collections import Counter

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    import numpy as np
    sklearn_available = True
except ImportError:
    sklearn_available = False
    print("sklearn not found, some features wont work properly")

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    colors_on = True
except ImportError:
    colors_on = False
    class Fore:
        RED = YELLOW = GREEN = CYAN = BLUE = WHITE = MAGENTA = ""
    class Style:
        RESET_ALL = ""


# ----------------------------------------------------------
# training data for priority classification
# basically gave it examples and labels, model learns from this
# ----------------------------------------------------------

training_sentences = [
    ("submit assignment deadline tonight", "HIGH"),
    ("urgent meeting with professor tomorrow", "HIGH"),
    ("pay electricity bill last date today", "HIGH"),
    ("exam preparation tomorrow morning", "HIGH"),
    ("fix critical bug in project", "HIGH"),
    ("hospital appointment urgent", "HIGH"),
    ("deadline project submission friday", "HIGH"),
    ("emergency task needed immediately", "HIGH"),
    ("interview preparation asap", "HIGH"),
    ("call doctor urgent", "HIGH"),
    ("complete report by end of day", "HIGH"),
    ("presentation due today", "HIGH"),

    ("buy groceries this week", "MEDIUM"),
    ("reply to emails", "MEDIUM"),
    ("read textbook chapter for class", "MEDIUM"),
    ("clean room before weekend", "MEDIUM"),
    ("prepare notes for next lecture", "MEDIUM"),
    ("update resume", "MEDIUM"),
    ("call friend to catch up", "MEDIUM"),
    ("research new laptop", "MEDIUM"),
    ("submit college application", "MEDIUM"),
    ("schedule dentist appointment", "MEDIUM"),
    ("organize study material", "MEDIUM"),
    ("revise previous chapters", "MEDIUM"),

    ("watch documentary sometime", "LOW"),
    ("try new restaurant", "LOW"),
    ("learn guitar when free", "LOW"),
    ("organise old photos", "LOW"),
    ("read book for fun", "LOW"),
    ("explore painting as hobby", "LOW"),
    ("rearrange furniture someday", "LOW"),
    ("update phone wallpaper", "LOW"),
    ("plan future vacation", "LOW"),
    ("watch cooking tutorial", "LOW"),
    ("try meditation eventually", "LOW"),
    ("download songs for playlist", "LOW"),
]

# words that hint at urgency - used to adjust prediction
urgent_words = ["urgent", "deadline", "asap", "today", "tonight", "critical",
                "emergency", "immediately", "must", "exam", "submit", "due"]

chill_words = ["someday", "eventually", "when free", "maybe", "fun",
               "relax", "watch", "explore", "plan future"]


def train_priority_model():
    if not sklearn_available:
        return None

    texts = [x[0] for x in training_sentences]
    labels = [x[1] for x in training_sentences]

    # using tfidf + naive bayes pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("nb", MultinomialNB(alpha=0.5))
    ])
    model.fit(texts, labels)
    return model


def get_urgency_score(text):
    # simple keyword check to help boost confidence
    t = text.lower()
    score = 0
    for w in urgent_words:
        if w in t:
            score += 1
    for w in chill_words:
        if w in t:
            score -= 1
    return score


priority_model = train_priority_model()


def predict_priority(text):
    # returns priority label and how confident the model is
    if priority_model is not None:
        probs = priority_model.predict_proba([text])[0]
        classes = priority_model.classes_
        boost = get_urgency_score(text)

        idx = {c: i for i, c in enumerate(classes)}

        # nudge probabilities based on keywords
        if boost > 0 and "HIGH" in idx:
            probs[idx["HIGH"]] = min(0.99, probs[idx["HIGH"]] + boost * 0.1)
        elif boost < 0 and "LOW" in idx:
            probs[idx["LOW"]] = min(0.99, probs[idx["LOW"]] + abs(boost) * 0.1)

        probs = probs / probs.sum()

        best = int(np.argmax(probs))
        label = classes[best]
        conf = round(probs[best] * 100, 1)

    else:
        # fallback if sklearn not installed
        boost = get_urgency_score(text)
        if boost >= 2:
            label, conf = "HIGH", 75.0
        elif boost == 1:
            label, conf = "HIGH", 60.0
        elif boost < 0:
            label, conf = "LOW", 65.0
        else:
            label, conf = "MEDIUM", 55.0

    return label, conf


# ----------------------------------------------------------
# completion prediction using logistic regression
# trained on some sample data i made manually
# features: word count, has deadline word, has action verb, priority level
# ----------------------------------------------------------

# [word_count, has_deadline, has_action_verb, priority_level, completed]
sample_data = [
    [3, 1, 1, 2, 1],
    [7, 1, 1, 2, 1],
    [4, 1, 1, 2, 1],
    [5, 0, 1, 1, 1],
    [6, 0, 1, 1, 1],
    [8, 1, 1, 1, 1],
    [4, 0, 0, 0, 0],
    [9, 0, 0, 0, 0],
    [12, 0, 0, 0, 0],
    [3, 0, 0, 0, 0],
    [10, 0, 1, 0, 0],
    [5, 1, 1, 2, 1],
    [6, 0, 0, 1, 0],
    [4, 1, 0, 1, 1],
    [7, 0, 1, 2, 1],
    [2, 1, 1, 2, 1],
    [15, 0, 0, 0, 0],
    [11, 0, 0, 0, 0],
]

deadline_hints = {"today", "tonight", "tomorrow", "deadline", "due",
                  "submit", "by", "asap", "urgent", "morning", "evening"}

action_verbs = {"buy", "call", "email", "write", "read", "complete",
                "finish", "send", "pay", "book", "schedule", "prepare"}


def get_completion_features(text, priority):
    words = text.lower().split()
    wc = len(words)
    has_deadline = int(any(w in deadline_hints for w in words))
    has_verb = int(any(w in action_verbs for w in words))
    pri_num = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}.get(priority, 1)
    return [wc, has_deadline, has_verb, pri_num]


def train_completion_model():
    if not sklearn_available:
        return None
    data = np.array(sample_data)
    X = data[:, :-1]
    y = data[:, -1]
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)
    return model


completion_model = train_completion_model()


def predict_completion(text, priority):
    features = get_completion_features(text, priority)

    if completion_model is not None:
        prob = completion_model.predict_proba([features])[0][1]
        pct = round(prob * 100, 1)
    else:
        score = 50
        score += features[1] * 15
        score += features[2] * 10
        score += features[3] * 5
        score -= max(0, features[0] - 8) * 2
        pct = float(max(5, min(95, score)))

    if pct >= 75:
        tip = "Good chance you'll finish this one."
    elif pct >= 50:
        tip = "Try setting a specific time to do this."
    elif pct >= 30:
        tip = "At risk of being forgotten. Set a reminder."
    else:
        tip = "Very vague task. Try making it more specific."

    return pct, tip


# ----------------------------------------------------------
# helper functions for display
# ----------------------------------------------------------

priority_colors = {
    "HIGH": Fore.RED,
    "MEDIUM": Fore.YELLOW,
    "LOW": Fore.GREEN
}

def clr(text, fore):
    return f"{fore}{text}{Style.RESET_ALL}"

def line(ch="-", n=60):
    print(clr(ch * n, Fore.BLUE))

def show_header():
    print()
    line("=")
    print(clr("   TO-DO LIST  -  Intelligent Task Manager", Fore.CYAN))
    line("=")
    print(clr("   Features: Priority Prediction + Completion Chance", Fore.WHITE))
    line()
    print()

def analyze_task(text):
    priority, conf = predict_priority(text)
    comp, tip = predict_completion(text, priority)
    return {
        "priority": priority,
        "confidence": conf,
        "completion": comp,
        "tip": tip
    }


# ----------------------------------------------------------
# main task list logic
# ----------------------------------------------------------

tasks = []


def show_tasks(mode=None):
    to_show = tasks

    if mode == "high":
        to_show = [t for t in tasks if t["info"]["priority"] == "HIGH"]
    elif mode == "done":
        to_show = [t for t in tasks if t["done"]]
    elif mode == "pending":
        to_show = [t for t in tasks if not t["done"]]

    if not to_show:
        print(clr("  No tasks found.", Fore.YELLOW))
        return

    line()
    for t in to_show:
        idx = tasks.index(t) + 1
        status = clr("Done", Fore.GREEN) if t["done"] else clr("Pending", Fore.YELLOW)
        print(f"  {clr(str(idx) + '.', Fore.CYAN)} [{status}]  {t['text']}")
    line()

    done = sum(1 for t in tasks if t["done"])
    print(clr(f"  Total: {len(tasks)}   Done: {done}   Pending: {len(tasks) - done}", Fore.CYAN))
    print()


def add_task():
    print(clr("\n  Enter tasks one by one. Type 'done' when finished.\n", Fore.WHITE))
    count = 0

    while True:
        user_input = input(clr(f"  Task {count + 1}: ", Fore.WHITE)).strip()

        if user_input.lower() in ("done", "quit", "exit"):
            break

        if user_input == "":
            print(clr("  Cant be empty. Type 'done' to stop.", Fore.YELLOW))
            continue

        print(clr("  Analyzing...", Fore.CYAN), end="", flush=True)
        info = analyze_task(user_input)
        print(clr(" done", Fore.GREEN))

        task = {
            "text": user_input,
            "done": False,
            "added": datetime.now().strftime("%d %b %Y %H:%M"),
            "info": info
        }
        tasks.append(task)
        count += 1

        pri = info["priority"]
        pc = priority_colors.get(pri, "")
        print(f"     Added  |  Priority: {clr(pri, pc)} ({info['confidence']}%)  |  "
              f"Completion chance: {clr(str(info['completion']) + '%', Fore.CYAN)}")
        print()

    if count > 0:
        line()
        print(clr(f"  {count} task(s) added.", Fore.GREEN))
        line()
    else:
        print(clr("  No tasks added.", Fore.YELLOW))


def delete_task():
    show_tasks()
    if not tasks:
        return
    try:
        n = int(input(clr("  Task number to delete: ", Fore.WHITE)))
        if 1 <= n <= len(tasks):
            removed = tasks.pop(n - 1)
            print(clr(f"  Removed: {removed['text']}", Fore.RED))
        else:
            print(clr("  Invalid number.", Fore.RED))
    except ValueError:
        print(clr("  Enter a valid number.", Fore.RED))


def mark_done():
    show_tasks()
    if not tasks:
        return
    try:
        n = int(input(clr("  Task number to mark as done: ", Fore.WHITE)))
        if 1 <= n <= len(tasks):
            tasks[n - 1]["done"] = True
            print(clr(f"  Marked done: {tasks[n-1]['text']}", Fore.GREEN))
        else:
            print(clr("  Invalid number.", Fore.RED))
    except ValueError:
        print(clr("  Enter a valid number.", Fore.RED))


def summary_report():
    if not tasks:
        print(clr("  No tasks yet.", Fore.YELLOW))
        return

    line("=")
    print(clr("  Summary Report", Fore.CYAN))
    line("=")

    pri_count = Counter(t["info"]["priority"] for t in tasks)

    print(clr("  Priority Breakdown:", Fore.WHITE))
    for p in ["HIGH", "MEDIUM", "LOW"]:
        bar = "#" * (pri_count.get(p, 0) * 4)
        print(f"    {clr(p.ljust(7), priority_colors.get(p, ''))}  {bar}  {pri_count.get(p, 0)}")

    avg = sum(t["info"]["completion"] for t in tasks) / len(tasks)
    print(clr(f"\n  Average Completion Chance: {round(avg, 1)}%", Fore.CYAN))

    risky = sorted(tasks, key=lambda t: t["info"]["completion"])[:3]
    print(clr("\n  Tasks least likely to be completed:", Fore.YELLOW))
    for t in risky:
        print(f"    - {t['text']}  ({clr(str(t['info']['completion']) + '%', Fore.RED)})")

    line("=")


# ----------------------------------------------------------
# main loop
# ----------------------------------------------------------

def main():
    show_header()

    menu = f"""
  {clr('1', Fore.CYAN)}. Add Task
  {clr('2', Fore.CYAN)}. View All Tasks
  {clr('3', Fore.CYAN)}. View High Priority Tasks
  {clr('4', Fore.CYAN)}. Mark Task as Done
  {clr('5', Fore.CYAN)}. Delete Task
  {clr('6', Fore.CYAN)}. Summary Report
  {clr('7', Fore.CYAN)}. Exit
"""

    while True:
        print(menu)
        choice = input(clr("  Enter choice: ", Fore.WHITE)).strip()

        if choice == "1":
            add_task()
        elif choice == "2":
            show_tasks()
        elif choice == "3":
            show_tasks(mode="high")
        elif choice == "4":
            mark_done()
        elif choice == "5":
            delete_task()
        elif choice == "6":
            summary_report()
        elif choice == "7":
            print(clr("\n  Bye!\n", Fore.CYAN))
            break
        else:
            print(clr("  Invalid. Enter 1 to 7.", Fore.RED))


if __name__ == "__main__":
    main()