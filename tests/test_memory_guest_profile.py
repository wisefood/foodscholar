"""A guest (or any member created without a profile) can be remembered.

Live: "Failed to apply memory for member 1a286d15…: Profile for household
member 1a286d15… not found". The gateway only PATCHes an existing profile.
"""

import unittest
from unittest.mock import MagicMock, patch

from services import memory_service
from services.memory_service import _writable_profile


class _Profile:
    def __init__(self, data):
        self._data = dict(data)

    def to_dict(self):
        return dict(self._data)


class _Member:
    def __init__(self, store):
        self.store = store
        self._profile = None

    @property
    def profile(self):
        if self._profile is None:
            self._profile = _Profile(self.store.get("profile") or {})
        return self._profile


class WritableProfileTests(unittest.TestCase):
    def _client(self, store):
        client = MagicMock()
        member = _Member(store)
        client.members.get.return_value = member

        def post(path, json=None):
            store["profile"] = {"id": "p1", "household_member_id": "m1", **(json or {})}
            store["posted"] = path

        client.post.side_effect = post
        return client

    def test_missing_profile_is_created_before_writing(self):
        store = {}
        profile = _writable_profile(self._client(store), "m1")
        self.assertEqual(store["posted"], "members/m1/profile")
        self.assertEqual(profile.to_dict()["id"], "p1")

    def test_existing_profile_is_not_recreated(self):
        store = {"profile": {"id": "p0"}}
        client = self._client(store)
        self.assertEqual(_writable_profile(client, "m1").to_dict()["id"], "p0")
        client.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
