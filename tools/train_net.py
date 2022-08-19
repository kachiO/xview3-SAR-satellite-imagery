#!/usr/bin/env python
"""
xView3 training script.
Adapted from https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py
"""

import ast
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch
from detectron2.evaluation import DatasetEvaluators, verify_results
from fvcore.nn.precise_bn import get_bn_modules
from xview3_d2.config.xview3 import add_xview3_config
from xview3_d2.data.datasets.xview3 import register_xview3_instances
from xview3_d2.data.xview3_build import (
    xview3_build_detection_test_loader,
    xview3_build_detection_train_loader,
)
from xview3_d2.engine.defaults import default_writers
from xview3_d2.engine.hooks import PeriodicWriter
from xview3_d2.evaluation.xview3_eval import xView3COCOEvaluator, xView3F1Evaluator

#register custom ROI Head
from xview3_d2.modeling.roi_heads.xview_custom_roi_heads import (
    xViewStandardROIHeads as _,
)

logger = logging.getLogger("detectron2.trainer")


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator for a xView3 dataset.
    """
    if output_folder is None:
        output_folder = (Path(cfg.OUTPUT_DIR) / f"inference_{dataset_name}").as_posix()

    evaluator_list = [
        xView3COCOEvaluator(dataset_name, output_dir=output_folder),
        xView3F1Evaluator(
            dataset_name,
            iou_thr=cfg.TEST.IOU_THRESHOLD,
            shoreline_dir=cfg.TEST.INPUT.SHORELINE_DIR,
            score_all=cfg.TEST.SCORE_ALL,
            score_thr=cfg.TEST.SCORE_THRESHOLD,
        ),
    ]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return xview3_build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return xview3_build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(
                hooks.BestCheckpointer(
                    cfg.TEST.EVAL_PERIOD, self.checkpointer, "aggregate"
                )
            )

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(PeriodicWriter(self.build_writers(), period=20))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter, filename_suffix='.tensorboard')


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_xview3_config(cfg)
    cfg.merge_from_file(args.config_file)
    config_fname = Path(args.config_file).stem
    opts = args.zopts[0].split(" ") if len(args.zopts) == 1 else args.zopts
    cfg.merge_from_list(opts)
    logger.info(f"opts: {opts}")

    if cfg.INPUT.DATA.SHORELINE_DIR is None:
        cfg.INPUT.DATA.SHORELINE_DIR = args.shoreline_dir
        cfg.TEST.INPUT.SHORELINE_DIR = args.shoreline_dir
        
    output_dir_base = Path(os.getenv('SM_MODEL_DIR', default=Path(cfg.OUTPUT_DIR).parent))
    timestamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
    output_dir = output_dir_base / config_fname / timestamp
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.EXP_NAME = f'{config_fname}-{timestamp}'
    os.environ['D2_OUTPUT_DIR'] = str(output_dir)
    os.environ['D2_EXP_NAME'] = cfg.EXP_NAME

    logger.info(f"OUTPUT_DIR: {cfg.OUTPUT_DIR}")

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def register_datasets(
    ds_name: str,
    dataset_dir: Union[os.PathLike, str],
    scene_imagery_dir: Optional[Union[os.PathLike, str]],
):
    """Register xView3 datasets from config file.

    Args:
        ds_name (str): dataset name from config.
        dataset_dir (Union[os.PathLike, str]): path to detectron2 dataset
        scene_imagery_dir (Union[os.PathLike, str]): path to scene imagery. if None provided, uses parent folder of `dataset_dir`.
    """
    assert Path(dataset_dir).exists(), f"{dataset_dir} does not exist."

    dataset_files = list(Path(dataset_dir).glob(f"{ds_name}*"))
    assert len(
        dataset_files
    ), f"No datasets found in {dataset_dir} matching {ds_name}. Please check directory."
    for ds_fname in dataset_files:
        register_xview3_instances(
            name=ds_fname.stem,
            dataset_file=ds_fname,
            scene_imagery_dir=scene_imagery_dir,
        )


def main(args):

    # extract files.
    if args.extract_data and comm.is_main_process():
        from utils import extract_files

        logger.info("Extracting .tar.gz files. This may take some time.")
        extract_files(
            files=Path(args.imagery_dir).glob("*tar.gz"),
            destination_dir=args.imagery_dir,
            delete_tar=True,
        )

    cfg = setup(args)

    # register train datasets.
    for dataset in set(cfg.DATASETS.TRAIN):
        register_datasets(
            ds_name=dataset,
            dataset_dir=args.d2_dataset_dir,
            scene_imagery_dir=args.imagery_dir,
        )
    # register validation/test datasets
    for dataset in set(cfg.DATASETS.TEST):
        register_datasets(
            ds_name=dataset,
            dataset_dir=args.d2_dataset_dir,
            scene_imagery_dir=args.valid_imagery_dir,
        )

    # run evaluation only
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def parse_args():
    from xview3_d2.engine.defaults import default_argument_parser

    parser = default_argument_parser()
    parser.add_argument(
        "--imagery-dir",
        default=os.getenv("SM_CHANNEL_IMAGERY"),
        required=True,
        help="imagery data directory",
    )
    parser.add_argument(
        "--valid-imagery-dir",
        default=os.getenv(
            "SM_CHANNEL_VALID_IMAGERY", default=os.getenv("SM_CHANNEL_IMAGERY")
        ),
        required=False,
        help="imagery data directory for validation set if different from `imagery-dir`",
    )
    parser.add_argument(
        "--d2-dataset-dir",
        default=os.getenv("SM_CHANNEL_DATASETS", default=None),
        help="Directory containing Detectron2-compatible dataset format.",
    )
    parser.add_argument(
        "--shoreline-dir",
        default=os.getenv("SM_CHANNEL_SHORELINE", default=None),
        help="Local directory containing shoreline data.",
    )
    parser.add_argument(
        "--extract-data",
        action="store_true",
        help="If data provided in `data-dir` are .tar.gz`, extract tar.gz files",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    logger.info(f"Command Line Args: {args}")
    machine_rank = args.machine_rank
    dist_url = args.dist_url
    num_machines = args.num_machines
    num_gpus = args.num_gpus

    if "SM_HOSTS" in os.environ:
        hosts = ast.literal_eval(os.environ["SM_HOSTS"])
        num_machines = len(hosts)
        machine_rank = hosts.index(os.environ["SM_CURRENT_HOST"])
        logger.info(f"Machine rank: {machine_rank}")
        master_addr = hosts[0]
        master_port = "55555"
        dist_url = "auto" if len(hosts) == 1 else f"tcp://{master_addr}:{master_port}"
        logger.info(f"Device URL: {dist_url}")
        num_gpus = ast.literal_eval(os.environ["SM_NUM_GPUS"])

    launch(
        main,
        num_gpus_per_machine=num_gpus,
        num_machines=num_machines,
        machine_rank=machine_rank,
        dist_url=dist_url,
        args=(args,),
    )
