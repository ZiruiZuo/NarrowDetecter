from __future__ import annotations

import unittest

from NarrowDetecter.gui.interactions import (
    ClickDisambiguator,
    HitTarget,
    MouseIntent,
    classify_pointer_action,
)


class _Scheduler:
    def __init__(self):
        self.callbacks = []

    def schedule(self, delay_ms, callback):
        token = [delay_ms, callback, False]
        self.callbacks.append(token)
        return token

    @staticmethod
    def cancel(token):
        token[2] = True

    def fire_all(self):
        for _, callback, cancelled in tuple(self.callbacks):
            if not cancelled:
                callback()


class InteractionTests(unittest.TestCase):
    def test_single_click_is_delayed_and_emitted_once(self):
        scheduler = _Scheduler()
        actions = []
        clicks = ClickDisambiguator(
            interval_ms=240,
            schedule=scheduler.schedule,
            cancel=scheduler.cancel,
        )

        clicks.release_empty((10, 20), actions.append, lambda: actions.append("max"))
        self.assertEqual(actions, [])
        scheduler.fire_all()
        self.assertEqual(actions, [(10, 20)])

    def test_second_click_cancels_append_and_maximizes_once(self):
        scheduler = _Scheduler()
        actions = []
        clicks = ClickDisambiguator(240, scheduler.schedule, scheduler.cancel)

        clicks.release_empty("point", actions.append, lambda: actions.append("max"))
        clicks.double_click(lambda: actions.append("max"))
        scheduler.fire_all()

        self.assertEqual(actions, ["max"])

    def test_only_active_point_can_start_drag_or_context_menu(self):
        active = HitTarget("active", 2, is_active=True)
        inactive = HitTarget("other", 0, is_active=False)

        self.assertEqual(
            classify_pointer_action("left", hit=active), MouseIntent.DRAG_POINT
        )
        self.assertEqual(
            classify_pointer_action("right", hit=active), MouseIntent.CONTEXT_MENU
        )
        self.assertEqual(
            classify_pointer_action("left", hit=inactive), MouseIntent.NONE
        )
        self.assertEqual(
            classify_pointer_action("right", hit=inactive), MouseIntent.NONE
        )

    def test_navigation_actions_are_unambiguous(self):
        self.assertEqual(classify_pointer_action("wheel"), MouseIntent.SLICE)
        self.assertEqual(
            classify_pointer_action("wheel", ctrl=True), MouseIntent.ZOOM
        )
        self.assertEqual(classify_pointer_action("middle"), MouseIntent.PAN)


if __name__ == "__main__":
    unittest.main()
