# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
from json import JSONDecodeError
from typing import Any

import datetime

import pandas

from annflux.tools.io import create_directory


class RepositoryObject(object):
    def __init__(self, name_value_cache_path):
        self.name_value_cache_path = name_value_cache_path

    def update_name_value_cache(self, name, value):
        """
        Writes name, value pairs to a basic_stats file
        :param name:
        :param value:
        :return:
        """
        j_cache = (
            json.load(open(self.name_value_cache_path))
            if os.path.exists(self.name_value_cache_path)
            else {}
        )

        j_cache[name] = value
        json.dump(j_cache, open(self.name_value_cache_path, "w"), indent=2)

    def get_cached_value(self, name):
        if not os.path.exists(self.name_value_cache_path):
            return None

        try:
            with open(self.name_value_cache_path) as f:
                j_cache = json.load(f)
        except JSONDecodeError:
            os.remove(self.name_value_cache_path)
            j_cache = {}

        return j_cache[name] if name in j_cache else None

    def cache(self, name, lazy_value):
        value = self.get_cached_value(name)
        if value is None:
            value = lazy_value()
            self.update_name_value_cache(name, value)
        return value


class RepositoryEntry(object):
    def __init__(self, row, repository):
        self.row = row
        self.repository_ = repository

    @property
    def path(self):
        return os.path.join(self.repository_.path, Repository.entry_to_path(self.row))

    @property
    def uid(self):
        return self.row[self.repository.headers.index("uid")]

    @property
    def label(self):
        return self.row[self.repository.headers.index("type")]

    @property
    def tag(self):
        return self.row[self.repository.headers.index("tag")]

    @property
    def date(self):
        return self.row[self.repository.headers.index("date")]

    @property
    def message(self):
        return self.row[self.repository.headers.index("message")]

    @property
    def ancestors(self):
        return self.row[self.repository.headers.index("ancestors")].split(";")

    @property
    def repository(self):
        return self.repository_

    def __eq__(self, right):
        return self.uid == right.uid


class CollectionWrapper(list):
    """
    Adds first and last methods to standard lists
    """

    def last(self, position=1):
        return self[-position] if len(self) > 0 else None

    def first(self):
        return self[0] if len(self) > 0 else None


class Repository(object):
    headers = ["uid", "type", "ancestors", "date", "tag", "message"]

    def __init__(self, path, date_filter=0):
        """
        Create a new Repository
        :param path: open or create Repository at path
        :param date_filter: experimental, obsolete
        """
        self.path = path
        self.index_path = os.path.join(path, "index.csv")
        self.date_filter = int(date_filter) if date_filter is not None else None
        if not os.path.exists(path):
            create_directory(path)
            # write_table(self.index_path, [], self.headers)
            pandas.DataFrame(data=[], columns=self.headers).to_csv(
                self.index_path, index=False
            )

    @staticmethod
    def entry_to_path(entry: RepositoryEntry):
        """
        Gives the folder name for a RepositoryEntry
        :param entry:
        :return:
        """
        # TODO: replace constant column indices by header lookup
        return "{}-{}-{}".format(entry[1], entry[3], entry[0][:8])  # noqa

    @staticmethod
    def get_ancestors(entry: RepositoryEntry, label: str, wrapper=None):
        ancestors = [x for x in entry.ancestors if label in x]
        if len(ancestors) > 1:
            raise Exception(
                "Number of ancestors of type {0} should be exactly 1 (uid = {1})".format(
                    label, entry.uid
                )
            )
        elif len(ancestors) == 0:
            return None
        repo = entry.repository_
        uid_full = ancestors[0]
        if ":" in uid_full:
            repo_path, uid_full = uid_full.split(":")
            repo = Repository(repo_path)
        ancestor_uid = uid_full.split("-")[1]
        entries = repo.get(uid=ancestor_uid)
        if len(entries) > 0:
            entry = entries[0]
        else:
            print("Missing entry for uid (dateFilter used?)", ancestor_uid)
            return None
        return wrapper(entry) if wrapper else entry

    def commit(
        self,
        obj: [Any, RepositoryObject],
        ancestors=None,
        tag="",
        message="",
        mode="move",
        allow_mixed_tags=False,
    ) -> [RepositoryObject, Any]:
        """
        Commits Repository objects so that they are versioned and persisted in a repository.
        Objects are related to each other through their ancestors.
        :param obj: object (KerasModel, DataSet, ResultSet) to be committed to the Repository
        :param ancestors: a list with elements of type RepositoryEntry or objects which have a property "entry" of type RepositoryEntry
        :param tag:
        :param message: commit message
        :param mode: obsolete, backwards compatibility
        :param allow_mixed_tags: experimental
        :return:
        """
        if ancestors is None:
            ancestors = []
        # TODO: backup index.csv at each commit to preempt corruption
        uid = obj.get_uid()
        existing_uids = [x.uid for x in self.entries]
        if uid in existing_uids:
            print(uid, "exists, ignore commit...")
            return self.get(uid=uid, label=obj.__class__).first()

        # resolve a possible mix of Repository objects and RepositoryEntry to a list of RepositoryEntry only
        ancestors_ = []
        for ancestor in ancestors:
            if isinstance(ancestor, RepositoryEntry):
                ancestors_.append(ancestor)
            else:
                if hasattr(ancestor, "entry"):
                    if isinstance(ancestor.entry, RepositoryEntry):
                        ancestors_.append(ancestor.entry)
                    else:
                        raise ValueError(
                            "ancestor.entry is not of type RepositoryEntry"
                        )
                else:
                    raise ValueError(
                        "ancestor is not of type RepositoryEntry and has no attribute entry"
                    )
        ancestors = ancestors_

        # by default allow_mixed_tags == False to prevent errors when committing
        if len(tag) > 0 and not allow_mixed_tags:
            ancestor_tags = [x.tag for x in ancestors]
            if ancestor_tags != [tag] * len(ancestor_tags):
                raise ValueError(
                    "one or more of the ancestor tags do not match supplied tag", tag, ancestor_tags
                )

        # by default allow_mixed_repos == False and objects must all be in the same repository to be committed
        for ancestor in ancestors:
            if ancestor.repository_.path != self.path:
                raise ValueError(
                    "one or more of the ancestors come from a different repository"
                )

        entry = [
            uid,
            obj.label,
            ";".join(
                [
                    (
                        "{}:".format(ancestor.path)
                        if ancestor.repository_.path != self.path
                        else ""
                    )
                    + ancestor.label
                    + "-"
                    + ancestor.uid
                    if type(ancestor) != str  # noqa
                    else ancestor
                    for ancestor in ancestors
                ]
            ),
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            tag,
            message,
        ]

        object_folder = create_directory(
            os.path.join(self.path, Repository.entry_to_path(entry))
        )

        obj.store_contents(object_folder, mode)

        # append_to_table(self.index_path, [entry])
        pandas.concat(
            [
                pandas.read_csv(self.index_path),
                pandas.DataFrame(data=[entry], columns=self.headers),
            ]
        ).to_csv(self.index_path, index=False)

        print("successfully committed", uid)

        return self.get(uid=uid, label=obj.__class__).first()

    @property
    def entries(self):
        table = pandas.read_csv(self.index_path, dtype={"date": str}).values
        # table, _ = read_table(self.index_path, return_as_dict=False)

        result = []
        for row in table:
            entry = RepositoryEntry(row, self)
            if self.date_filter is None:
                result.append(entry)
            elif int(entry.date[:8]) > self.date_filter:
                result.append(entry)
        return result

    @property
    def resultsets(self):
        return [x for x in self.entries if x.label == "resultset"]

    @property
    def models(self):
        return [x for x in self.entries if x.label == "model"]

    @property
    def datasets(self):
        return [x for x in self.entries if x.label == "dataset"]

    def get(self, label=None, tag=None, uid=None):
        obj = None
        if label is not None and not isinstance(label, str):
            obj = label
            label = obj.label
        result = [
            x
            for x in self.entries
            if (x.label == label or label is None)
            and (x.tag == tag or tag is None)
            and (uid is None or x.uid.startswith(uid))
        ]
        if uid is not None and len(result) > 1:
            raise Exception("Multiple results for shorthand uid {}".format(uid))
        if obj is not None:
            result = list(map(obj, result))
        return CollectionWrapper(result)

    @property
    def size(self):
        return len(pandas.read_csv(self.index_path))
