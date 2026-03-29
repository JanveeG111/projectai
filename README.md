# projectai
This is a to-do list application I created for my AIML project. It's not just a regular task manager – I added machine learning to predict which tasks are important and which ones you'll probably forget about. 
# Smart To-Do List with ML-Based Priority Prediction

A command-line to-do list application built in Python that uses machine learning to automatically predict the **priority** of each task you add and estimate the **likelihood you'll actually complete it**.

This was made as part of an AI/ML course project. The core idea was to take a basic to-do list and make it smarter using real ML models instead of just hardcoded rules.


## What It Does

When you add a task, the app runs it through two ML models:

- **Priority Prediction** — Uses a Naive Bayes classifier trained on TF-IDF features to label your task as `HIGH`, `MEDIUM`, or `LOW` priority. It also uses keyword boosting on top of the model to improve accuracy for obvious cases like deadlines and urgent tasks.

- **Completion Chance** — Uses a Logistic Regression model trained on task features (word count, whether the task has a deadline word, whether it has a clear action verb, and the priority level) to estimate the percentage chance that you'll actually finish the task. It also gives a short tip based on the score.

Everything runs locally, no internet needed.


## Features

- Add multiple tasks in one go — just keep typing until you're done
- Each task gets auto-analysed for priority and completion chance right after you enter it
- View all your tasks in a clean numbered list
- Filter to only see high priority tasks
- Mark tasks as done
- Delete tasks
- Summary report showing priority breakdown and which tasks are most at risk of being forgotten


## Project Structure
todo-ml/
│
├── ai_todo_list.py      # main file, run this
└── README.md


All the ML code (training data, model training, prediction functions) is inside `ai_todo_list.py`. No separate files needed.

## Requirements

- Python 3.7 or above
- The following libraries:

| Library | What it's used for |
|---|---|
| `scikit-learn` | Naive Bayes and Logistic Regression models |
| `numpy` | Array operations for model training |
| `colorama` | Coloured terminal output |

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/your-username/todo-ml.git
cd todo-ml
```

**2. Install the dependencies**

```bash
pip install scikit-learn numpy colorama
```

> If you're using Python 3 and `pip` doesn't work, try `pip3` instead.

**3. Run the app**

```bash
python ai_todo_list.py
```

> On some systems you may need to use `python3 ai_todo_list.py`

## How to Use

When you run the file you'll see a menu like this:

```
============================================================
   TO-DO LIST  -  Intelligent Task Manager
============================================================
   Features: Priority Prediction + Completion Chance
------------------------------------------------------------

  1. Add Task
  2. View All Tasks
  3. View High Priority Tasks
  4. Mark Task as Done
  5. Delete Task
  6. Summary Report
  7. Exit
```

### Adding Tasks

Choose option `1`. The app will ask you to enter tasks one by one. Type each task and press Enter. When you're done entering all your tasks, type `done` and press Enter.

```
  Enter tasks one by one. Type 'done' when finished.

  Task 1: submit machine learning assignment by tonight
  Analyzing... done
     Added  |  Priority: HIGH (92.3%)  |  Completion chance: 81.0%

  Task 2: watch a movie sometime
  Analyzing... done
     Added  |  Priority: LOW (88.1%)  |  Completion chance: 34.5%

  Task 3: done
```

### Viewing Tasks

Option `2` shows all your tasks with their status (Pending / Done).  
Option `3` filters and shows only HIGH priority tasks.

### Marking Done / Deleting

Both options first show your task list so you can see the numbers, then ask which task number to act on.

### Summary Report

Option `6` gives an overview across all your tasks:

```
  Priority Breakdown:
    HIGH     ####  1
    MEDIUM   ############  3
    LOW      ####  1

  Average Completion Chance: 61.4%

  Tasks least likely to be completed:
    - watch a movie sometime  (34.5%)
```

---

## How the ML Works

### Priority Prediction (Naive Bayes + TF-IDF)

The training data is a list of 36 labelled example sentences covering HIGH, MEDIUM, and LOW priority tasks. At startup, the app trains a `Pipeline` combining `TfidfVectorizer` (converts text to numerical features using word frequency) and `MultinomialNB` (Naive Bayes classifier).

When you enter a task, it gets passed through this pipeline and the model outputs a predicted label along with confidence probabilities. A keyword boost step then slightly adjusts the probabilities if obvious urgency words (like `deadline`, `asap`, `tonight`) or low-priority words (like `someday`, `eventually`) are detected.

### Completion Prediction (Logistic Regression)

Four features are extracted from each task:

1. Word count
2. Whether the task contains a deadline-related word
3. Whether the task contains a clear action verb (buy, call, submit, etc.)
4. The predicted priority level (encoded as 0, 1, 2)

These features are fed into a `LogisticRegression` model trained on 18 manually labelled samples. The model outputs a probability score (0–100%) which is shown as the completion chance.

---

## Notes

- The models are trained fresh each time you run the app. Since the training data is small and fixed, this only takes a fraction of a second.
- If `scikit-learn` is not installed, the app falls back to a simpler keyword-based system so it still works, just less accurately.
- If `colorama` is not installed, the app still runs fine but without colours.
- Task data is not saved between sessions. Closing the app will clear all tasks.

---

## Example Run

```
  Task 1: pay phone bill today
  Analyzing... done
     Added  |  Priority: HIGH (89.5%)  |  Completion chance: 78.0%

  Task 2: learn to cook pasta someday
  Analyzing... done
     Added  |  Priority: LOW (91.2%)  |  Completion chance: 29.5%

  Task 3: prepare slides for tomorrow presentation
  Analyzing... done
     Added  |  Priority: HIGH (85.0%)  |  Completion chance: 72.0%

  Task 4: done
  ------------------------------------------------------------
  3 task(s) added.
```

---

## Dependencies Version Reference

Tested with:

```
Python        3.10+
scikit-learn  1.3.0
numpy         1.24.0
colorama      0.4.6
```

Older versions should work fine too as long as Python is 3.7+.
