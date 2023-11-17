"""This file contains code to build dataloader of COCO-split dataset.

Reference:
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""

import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
# Added for cross-category evaluation
from .cocoeval_wrappers import COCOEvalXclassWrapper

from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class Objects365SplitDataset(CocoDataset):

    def __init__(self, 
                 is_class_agnostic=False, 
                 train_class='all',
                 eval_class='all',
                 **kwargs):
        # We convert all category IDs into 1 for the class-agnostic training and
        # evaluation. We train on train_class and evaluate on eval_class split.
        self.is_class_agnostic = is_class_agnostic
        self.train_class = train_class
        self.eval_class = eval_class
        super(Objects365SplitDataset, self).__init__(**kwargs)
        self.dataset_stat()
    
    CLASSES = ("Person", "Sneakers", "Chair", "Other Shoes", "Hat", "Car", "Lamp", "Glasses", "Bottle", "Desk", "Cup", "Street Lights", "Cabinet/shelf", "Handbag/Satchel", "Bracelet", "Plate", "Picture/Frame", "Helmet", "Book", "Gloves", "Storage box", "Boat", "Leather Shoes", "Flower", "Bench", "Potted Plant", "Bowl/Basin", "Flag", "Pillow", "Boots", "Vase", "Microphone", "Necklace", "Ring", "SUV", "Wine Glass", "Belt", "Moniter/TV", "Backpack", "Umbrella", "Traffic Light", "Speaker", "Watch", "Tie", "Trash bin Can", "Slippers", "Bicycle", "Stool", "Barrel/bucket", "Van", "Couch", "Sandals", "Bakset", "Drum", "Pen/Pencil", "Bus", "Wild Bird", "High Heels", "Motorcycle", "Guitar", "Carpet", "Cell Phone", "Bread", "Camera", "Canned", "Truck", "Traffic cone", "Cymbal", "Lifesaver", "Towel", "Stuffed Toy", "Candle", "Sailboat", "Laptop", "Awning", "Bed", "Faucet", "Tent", "Horse", "Mirror", "Power outlet", "Sink", "Apple", "Air Conditioner", "Knife", "Hockey Stick", "Paddle", "Pickup Truck", "Fork", "Traffic Sign", "Ballon", "Tripod", "Dog", "Spoon", "Clock", "Pot", "Cow", "Cake", "Dinning Table", "Sheep", "Hanger", "Blackboard/Whiteboard", "Napkin", "Other Fish", "Orange/Tangerine", "Toiletry", "Keyboard", "Tomato", "Lantern", "Machinery Vehicle", "Fan", "Green Vegetables", "Banana", "Baseball Glove", "Airplane", "Mouse", "Train", "Pumpkin", "Soccer", "Skiboard", "Luggage", "Nightstand", "Tea pot", "Telephone", "Trolley", "Head Phone", "Sports Car", "Stop Sign", "Dessert", "Scooter", "Stroller", "Crane", "Remote", "Refrigerator", "Oven", "Lemon", "Duck", "Baseball Bat", "Surveillance Camera", "Cat", "Jug", "Broccoli", "Piano", "Pizza", "Elephant", "Skateboard", "Surfboard", "Gun", "Skating and Skiing shoes", "Gas stove", "Donut", "Bow Tie", "Carrot", "Toilet", "Kite", "Strawberry", "Other Balls", "Shovel", "Pepper", "Computer Box", "Toilet Paper", "Cleaning Products", "Chopsticks", "Microwave", "Pigeon", "Baseball", "Cutting/chopping Board", "Coffee Table", "Side Table", "Scissors", "Marker", "Pie", "Ladder", "Snowboard", "Cookies", "Radiator", "Fire Hydrant", "Basketball", "Zebra", "Grape", "Giraffe", "Potato", "Sausage", "Tricycle", "Violin", "Egg", "Fire Extinguisher", "Candy", "Fire Truck", "Billards", "Converter", "Bathtub", "Wheelchair", "Golf Club", "Briefcase", "Cucumber", "Cigar/Cigarette ", "Paint Brush", "Pear", "Heavy Truck", "Hamburger", "Extractor", "Extention Cord", "Tong", "Tennis Racket", "Folder", "American Football", "earphone", "Mask", "Kettle", "Tennis", "Ship", "Swing", "Coffee Machine", "Slide", "Carriage", "Onion", "Green beans", "Projector", "Frisbee", "Washing Machine/Drying Machine", "Chicken", "Printer", "Watermelon", "Saxophone", "Tissue", "Toothbrush", "Ice cream", "Hotair ballon", "Cello", "French Fries", "Scale", "Trophy", "Cabbage", "Hot dog", "Blender", "Peach", "Rice", "Wallet/Purse", "Volleyball", "Deer", "Goose", "Tape", "Tablet", "Cosmetics", "Trumpet", "Pineapple", "Golf Ball", "Ambulance", "Parking meter", "Mango", "Key", "Hurdle", "Fishing Rod", "Medal", "Flute", "Brush", "Penguin", "Megaphone", "Corn", "Lettuce", "Garlic", "Swan", "Helicopter", "Green Onion", "Sandwich", "Nuts", "Speed Limit Sign", "Induction Cooker", "Broom", "Trombone", "Plum", "Rickshaw", "Goldfish", "Kiwi fruit", "Router/modem", "Poker Card", "Toaster", "Shrimp", "Sushi", "Cheese", "Notepaper", "Cherry", "Pliers", "CD", "Pasta", "Hammer", "Cue", "Avocado", "Hamimelon", "Flask", "Mushroon", "Screwdriver", "Soap", "Recorder", "Bear", "Eggplant", "Board Eraser", "Coconut", "Tape Measur/ Ruler", "Pig", "Showerhead", "Globe", "Chips", "Steak", "Crosswalk Sign", "Stapler", "Campel", "Formula 1 ", "Pomegranate", "Dishwasher", "Crab", "Hoverboard", "Meat ball", "Rice Cooker", "Tuba", "Calculator", "Papaya", "Antelope", "Parrot", "Seal", "Buttefly", "Dumbbell", "Donkey", "Lion", "Urinal", "Dolphin", "Electric Drill", "Hair Dryer", "Egg tart", "Jellyfish", "Treadmill", "Lighter", "Grapefruit", "Game board", "Mop", "Radish", "Baozi", "Target", "French", "Spring Rolls", "Monkey", "Rabbit", "Pencil Case", "Yak", "Red Cabbage", "Binoculars", "Asparagus", "Barbell", "Scallop", "Noddles", "Comb", "Dumpling", "Oyster", "Table Teniis paddle", "Cosmetics Brush/Eyeliner Pencil", "Chainsaw", "Eraser", "Lobster", "Durian", "Okra", "Lipstick", "Cosmetics Mirror", "Curling", "Table Tennis ")
    COCO_CLASSES = ("Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat", "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking meter", "Bench", "Wild Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag/Satchel", "Tie", "Frisbee", "Skiboard", "Snowboard", "Kite", "Baseball Bat", "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle", "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl/Basin", "Banana", "Apple", "Sandwich", "Orange/Tangerine", "Broccoli", "Carrot", "Hot dog", "Pizza", "Donut", "Cake", "Chair", "Couch", "Potted Plant", "Bed", "Dinning Table", "Toilet", "Moniter/TV", "Laptop", "Mouse", "Remote", "Keyboard", "Cell Phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Hair Dryer", "Toothbrush", "Baseball", "Basketball", "American Football")
    NONCOCO_CLASSES = ("Sneakers", "Other Shoes", "Hat", "Lamp", "Glasses", "Desk", "Street Lights", "Cabinet/shelf", "Bracelet", "Plate", "Picture/Frame", "Helmet", "Gloves", "Storage box", "Leather Shoes", "Flower", "Flag", "Pillow", "Boots", "Microphone", "Necklace", "Ring", "SUV", "Belt", "Speaker", "Watch", "Trash bin Can", "Slippers", "Stool", "Barrel/bucket", "Van", "Sandals", "Bakset", "Drum", "Pen/Pencil", "High Heels", "Guitar", "Carpet", "Bread", "Camera", "Canned", "Traffic cone", "Cymbal", "Lifesaver", "Towel", "Stuffed Toy", "Candle", "Sailboat", "Awning", "Faucet", "Tent", "Mirror", "Power outlet", "Air Conditioner", "Hockey Stick", "Paddle", "Pickup Truck", "Traffic Sign", "Ballon", "Tripod", "Pot", "Hanger", "Blackboard/Whiteboard", "Napkin", "Other Fish", "Toiletry", "Tomato", "Lantern", "Machinery Vehicle", "Fan", "Green Vegetables", "Pumpkin", "Soccer", "Luggage", "Nightstand", "Tea pot", "Telephone", "Trolley", "Head Phone", "Sports Car", "Dessert", "Scooter", "Stroller", "Crane", "Lemon", "Duck", "Surveillance Camera", "Jug", "Piano", "Gun", "Skating and Skiing shoes", "Gas stove", "Bow Tie", "Strawberry", "Other Balls", "Shovel", "Pepper", "Computer Box", "Toilet Paper", "Cleaning Products", "Chopsticks", "Pigeon", "Cutting/chopping Board", "Coffee Table", "Side Table", "Marker", "Pie", "Ladder", "Cookies", "Radiator", "Grape", "Potato", "Sausage", "Tricycle", "Violin", "Egg", "Fire Extinguisher", "Candy", "Fire Truck", "Billards", "Converter", "Bathtub", "Wheelchair", "Golf Club", "Briefcase", "Cucumber", "Cigar/Cigarette ", "Paint Brush", "Pear", "Heavy Truck", "Hamburger", "Extractor", "Extention Cord", "Tong", "Folder", "earphone", "Mask", "Kettle", "Tennis", "Ship", "Swing", "Coffee Machine", "Slide", "Carriage", "Onion", "Green beans", "Projector", "Washing Machine/Drying Machine", "Chicken", "Printer", "Watermelon", "Saxophone", "Tissue", "Ice cream", "Hotair ballon", "Cello", "French Fries", "Scale", "Trophy", "Cabbage", "Blender", "Peach", "Rice", "Wallet/Purse", "Volleyball", "Deer", "Goose", "Tape", "Tablet", "Cosmetics", "Trumpet", "Pineapple", "Golf Ball", "Ambulance", "Mango", "Key", "Hurdle", "Fishing Rod", "Medal", "Flute", "Brush", "Penguin", "Megaphone", "Corn", "Lettuce", "Garlic", "Swan", "Helicopter", "Green Onion", "Nuts", "Speed Limit Sign", "Induction Cooker", "Broom", "Trombone", "Plum", "Rickshaw", "Goldfish", "Kiwi fruit", "Router/modem", "Poker Card", "Shrimp", "Sushi", "Cheese", "Notepaper", "Cherry", "Pliers", "CD", "Pasta", "Hammer", "Cue", "Avocado", "Hamimelon", "Flask", "Mushroon", "Screwdriver", "Soap", "Recorder", "Eggplant", "Board Eraser", "Coconut", "Tape Measur/ Ruler", "Pig", "Showerhead", "Globe", "Chips", "Steak", "Crosswalk Sign", "Stapler", "Campel", "Formula 1 ", "Pomegranate", "Dishwasher", "Crab", "Hoverboard", "Meat ball", "Rice Cooker", "Tuba", "Calculator", "Papaya", "Antelope", "Parrot", "Seal", "Buttefly", "Dumbbell", "Donkey", "Lion", "Urinal", "Dolphin", "Electric Drill", "Egg tart", "Jellyfish", "Treadmill", "Lighter", "Grapefruit", "Game board", "Mop", "Radish", "Baozi", "Target", "French", "Spring Rolls", "Monkey", "Rabbit", "Pencil Case", "Yak", "Red Cabbage", "Binoculars", "Asparagus", "Barbell", "Scallop", "Noddles", "Comb", "Dumpling", "Oyster", "Table Teniis paddle", "Cosmetics Brush/Eyeliner Pencil", "Chainsaw", "Eraser", "Lobster", "Durian", "Okra", "Lipstick", "Cosmetics Mirror", "Curling", "Table Tennis ")

    class_names_dict = {
        'all': CLASSES,
        'coco': COCO_CLASSES,
        'noncoco': NONCOCO_CLASSES
    }

    def dataset_stat(self):
        num_images = len(self)
        num_instances = 0
        for i in range(num_images):
            ann = self.get_ann_info(i)
            num_bbox = ann['bboxes'].shape[0]
            num_instances += num_bbox
        print(f'Dataset images number: {num_images}')
        print(f'Dataset instances number: {num_instances}')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)

        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.train_cat_ids = self.coco.get_cat_ids(
            cat_names=self.class_names_dict[self.train_class]
            )
        self.eval_cat_ids = self.coco.get_cat_ids(
            cat_names=self.class_names_dict[self.eval_class]
            )
        if self.is_class_agnostic:
            self.cat2label = {cat_id: 0 for cat_id in self.cat_ids}
        else:
            self.cat2label = {
                # cat_id: i for i, cat_id in enumerate(self.cat_ids)}
                cat_id: i for i, cat_id in enumerate(self.train_cat_ids)}

        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            # for objects 365
            # info['file_name'] = info['file_name'].split('/', 2)[-1]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    # Refer to custom.py -- filter_img is not used in test_mode.
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        
        for i, class_id in enumerate(self.train_cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.train_cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)                
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(10, 20, 30, 50, 100, 300, 500, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO-Split protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                # results = [ele[0] for ele in results] # for agnostic detection
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric

            # Class manipulation.
            for idx, ann in enumerate(cocoGt.dataset['annotations']):
                if ann['category_id'] in self.eval_cat_ids:
                    cocoGt.dataset['annotations'][idx]['ignored_split'] = 0
                else:
                    cocoGt.dataset['annotations'][idx]['ignored_split'] = 1

            # Cross-category evaluation wrapper.
            cocoEval = COCOEvalXclassWrapper(cocoGt, cocoDt, iou_type)

            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'AR_50': 1,
                'AR_75': 2,
                'AR_s': 3,
                'AR_m': 4,
                'AR_l': 5,
                'AR@10': 6,
                'AR@20': 7,
                'AR@30': 8,
                'AR@50': 9,
                'AR@100': 10,
                'AR@300': 11,
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            cocoEval.params.useCats = 0  # treat all FG classes as single class.
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'AR_50', 'AR_75', 'AR_s', 'AR_m', 'AR_l', 
                    'AR@10', 'AR@30', 'AR@50', 'AR@100', 'AR@300'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEval.stats[:3]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
