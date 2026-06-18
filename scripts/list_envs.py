# This file contains code adapted from https://github.com/unitreerobotics/unitree_rl_mjlab
# Original Project Copyright 2026 Unitree
# Original Project License: Apache License 2.0
#
# --------------------------------------------------------------------------
# Modifications Copyright 2026 VinRobotics
#
# This file has been modified. Changes and additions are licensed under  Apache 2.0

"""Script to list mjlab environments."""

import mjlab
import mjlab.tasks  # noqa: F401
import tyro
from mjlab.tasks.registry import list_tasks
from prettytable import PrettyTable
import src.tasks  # noqa: F401


def list_environments(keyword: str | None = None):
  """List all registered environments.

  Args:
    keyword: Optional filter to only show environments containing this keyword.
  """
  table = PrettyTable(["#", "Task ID"])
  table.title = "Available Environments in mjlab"
  table.align["Task ID"] = "l"

  all_tasks = list_tasks()
  idx = 0
  for task_id in all_tasks:
    try:
      # Optionally filter by keyword.
      if keyword and keyword.lower() not in task_id.lower():
        continue

      table.add_row([idx + 1, task_id])
      idx += 1
    except (AttributeError, TypeError):
      continue

  print(table)
  if idx == 0:
    msg = "[INFO] No tasks matched"
    if keyword:
      msg += f" keyword '{keyword}'"
    print(msg)
  return idx


def main():
  return tyro.cli(list_environments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
