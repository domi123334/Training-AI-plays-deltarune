"""
priority_actions.py

Fixed party-management rules that take priority over the normal menu
advisor pick, based on party HP and TP state:

  1. If any party member's HP is critical (yellow number) and ITEM is
     available this turn, go heal with the Dark Burger.
  2. Otherwise, if TP is above TP_THRESHOLD (82%): prefer Ralsei's Heal
     Prayer (via ACT) if the party is low on HP, else Susie's Rude Buster
     (via FIGHT) for a big damage swing.
  3. Otherwise, defer (return None) to the normal advisor/heuristic pick.

IMPORTANT NOTE ON TP: Deltarune's special moves have a FIXED TP cost, not
an arbitrary allocation - there's no way to "send 32% of TP" to one move
and "50%" to another the way a resource-allocation system would. You
either have enough TP for a move or you don't. The percentages here gate
WHEN each move is chosen (only once TP is plentiful enough that using it
is a good trade), not how much of it gets spent - actual TP cost is
whatever Heal Prayer/Rude Buster costs in-game, which this doesn't (and
can't, without memory access) verify directly.
"""

from typing import Optional

TP_THRESHOLD = 0.82

ITEM_TARGET_TEXT = "DARK BURGER"
HEAL_PRAYER_TARGET_TEXT = "HEAL PRAYER"
RUDE_BUSTER_TARGET_TEXT = "RUDE BUSTER"


class PriorityDecisionMaker:
    def decide_top_level(
        self, low_hp_chars: list[str], tp_fraction: float, visible_labels: list[str]
    ) -> Optional[str]:
        """
        Returns one of the labels from visible_labels to force-pick this
        turn, or None to defer to the normal advisor/heuristic pick.
        """
        upper_labels = [label.upper() for label in visible_labels]

        if low_hp_chars and any("ITEM" in label for label in upper_labels):
            return next(label for label in visible_labels if "ITEM" in label.upper())

        if tp_fraction >= TP_THRESHOLD:
            if low_hp_chars and any("ACT" in label for label in upper_labels):
                return next(label for label in visible_labels if "ACT" in label.upper())
            if any("FIGHT" in label for label in upper_labels):
                return next(label for label in visible_labels if "FIGHT" in label.upper())

        return None

    def target_text_for(self, top_level_label: str) -> Optional[str]:
        """
        Given the top-level verb this decision maker forced, returns the
        specific submenu option text to look for next (e.g. after ITEM,
        look for "DARK BURGER"). Returns None if this verb wasn't one of
        ours (i.e. the normal advisor picked it, not a priority rule).
        """
        label = top_level_label.upper()
        if "ITEM" in label:
            return ITEM_TARGET_TEXT
        if "ACT" in label:
            return HEAL_PRAYER_TARGET_TEXT
        if "FIGHT" in label:
            return RUDE_BUSTER_TARGET_TEXT
        return None
