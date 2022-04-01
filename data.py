import json
import os
import os.path
import re

import cv2
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import utils.utils as utils
from PIL import Image

with open("config.json", "r") as conf:
    config = json.loads(conf.read())


def get_loader(train=False, val=False, test=False):
    """Returns a data loader for the desired split"""
    assert (
        train + val + test == 1
    ), "need to set exactly one of {train, val, test} to True"
    split = VQA(
        utils.path_for(train=train, val=val, test=test, question=True),
        utils.path_for(train=train, val=val, test=test, answer=True),
        config.get("preprocessed_path"),
        answerable_only=train,
    )
    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config.get("batch_size"),
        shuffle=train,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.get("data_workers"),
        collate_fn=collate_fn,
    )
    return loader


def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)


class CocoImages(data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.id_to_filename = self.find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())
        print("found {} images in {}".format(len(self), self.path))
        self.transform = transform

    def find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith(".jpg"):
                continue
            id_and_extension = filename.split("_")[-1]
            id = int(id_and_extension.split(".")[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)


class Composite(data.Dataset):
    """Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset."""

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        current = self.datasets[0]
        for d in self.datasets:
            if item < len(d):
                return d[item]
            item -= len(d)
        else:
            raise IndexError("Index too large for composite dataset")

    def __len__(self):
        return sum(map(len, self.datasets))


class VQA(data.Dataset):
    def __init__(
        self,
        questions_path,
        answers_path,
        image_features_path,
        answerable_only=False,
    ):
        super().__init__()
        with open(config.get("question_train"), "r") as f:
            questions_json = json.load(f)
        with open(config.get("answer_train"), "r") as f:
            answers_json = json.load(f)
        with open(config.get("vocab_path"), "r") as fd:
            vocab_json = json.load(fd)
        self.check_integrity(questions_json, answers_json)

        # vocab
        self.vocab = vocab_json
        self.question_vocab = self.vocab["question_vocab"]
        self.answer_vocab = self.vocab["answer_vocab"]

        # q and a
        self.questions = list(utils.question_loading(questions_json))
        self.answers = list(utils.answer_loading(answers_json))
        self.questions = [self.encode_questions(q) for q in self.questions]
        self.answers = [self.encode_answers(a) for a in self.answers]

        # v
        self.image_features_path = image_features_path
        self.coco_id_to_index = self._create_coco_id_to_index()
        self.coco_ids = [q["image_id"] for q in questions_json["questions"]]

        # only use questions that have at least one answer?
        self.answerable_only = answerable_only
        if self.answerable_only:
            self.answerable = self._find_answerable()

    def __len__(self):
        if self.answerable_only:
            return len(self.answerable)
        else:
            return len(self.questions)

    def _find_answerable(self):
        """Create a list of indices into questions that will have at least one answer that is in the vocab"""
        answerable = []
        for i, answers in enumerate(self.answers):
            answer_has_index = len(answers.nonzero()) > 0
            # store the indices of anything that is answerable
            if answer_has_index:
                answerable.append(i)
        return answerable

    @property
    def max_question_length(self):
        if not hasattr(self, "_max_length"):
            self._max_length = max(map(len, self.questions))
        return self._max_length

    @property
    def num_tokens(self):
        return len(self.question_vocab) + 1  # 1 for unknow questions

    def encode_questions(self, question):
        tokens = torch.zeros(self.max_question_length).long()
        for i, token in enumerate(question):
            index = self.question_vocab.get(token, 0)
            tokens[i] = index
        return tokens, len(question)

    def encode_answers(self, answers):
        answer_vec = torch.zeros(len(self.answer_vocab))
        for answer in answers:
            index = self.answer_vocab.get(answer)
            if index is not None:
                answer_vec[index] += 1
        return answer_vec

    def check_integrity(self, questions, answers):
        qa_pairs = list(zip(questions["questions"], answers["annotations"]))
        assert all(
            q["question_id"] == a["question_id"] for q, a in qa_pairs
        ), "Questions not aligned with answers"
        assert all(
            q["image_id"] == a["image_id"] for q, a in qa_pairs
        ), "Image id of question and answer don't match"
        assert questions["data_type"] == answers["data_type"], "Mismatched data types"
        assert (
            questions["data_subtype"] == answers["data_subtype"]
        ), "Mismatched data subtypes"

    def _create_coco_id_to_index(self):
        """Create a mapping from a COCO image id into the corresponding index into the h5 file"""
        with h5py.File(self.image_features_path, "r") as features_file:
            coco_ids = features_file["ids"][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def load_image(self, image_id):
        """Load an image"""
        if not hasattr(self, "features_file"):
            self.features_file = h5py.File(self.image_features_path, "r")
        index = self.coco_id_to_index[image_id]
        dataset = self.features_file["features"]
        img = dataset[index].astype("float32")
        return torch.from_numpy(img)

    def __getitem__(self, item):
        if self.answerable_only:
            item = self.answerable[item]
        q, q_length = self.questions[item]
        a = self.answers[item]
        image_id = self.coco_ids[item]
        v = self.load_image(image_id)
        return v, q, a, item, q_length
