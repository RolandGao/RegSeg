import os
from PIL import Image
import numpy as np
import glob
from cityscapesscripts.helpers.labels import trainId2label
def save_results(pred_dir,file_name, output):
    basename = os.path.splitext(os.path.basename(file_name))[0]
    pred_filename = os.path.join(pred_dir, basename + "_pred.png")

    output = output.cpu().numpy()
    pred = 255 * np.ones(output.shape, dtype=np.uint8)
    for train_id, label in trainId2label.items():
        if label.ignoreInEval:
            continue
        pred[output == train_id] = label.id
    Image.fromarray(pred).save(pred_filename)

# https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/cityscapes_evaluation.html
def evaluate_cityscapes(pred_dir,gt_dir):
    # Load the Cityscapes eval script *after* setting the required env var,
    # since the script reads CITYSCAPES_DATASET into global variables at load time.
    import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

    # set some global states in cityscapes evaluation API, before evaluating
    cityscapes_eval.args.predictionPath = os.path.abspath(pred_dir)
    cityscapes_eval.args.predictionWalk = None
    cityscapes_eval.args.JSONOutput = False
    cityscapes_eval.args.colorized = False

    # These lines are adopted from
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
    groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
    assert len(
        groundTruthImgList
    ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
        cityscapes_eval.args.groundTruthSearch
    )
    predictionImgList = []
    for gt in groundTruthImgList:
        predictionImgList.append(cityscapes_eval.getPrediction(cityscapes_eval.args, gt))
    results = cityscapes_eval.evaluateImgLists(
        predictionImgList, groundTruthImgList, cityscapes_eval.args
    )
    ret = {}
    ret["sem_seg"] = {
        "IoU": 100.0 * results["averageScoreClasses"],
        "iIoU": 100.0 * results["averageScoreInstClasses"],
        "IoU_sup": 100.0 * results["averageScoreCategories"],
        "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
    }
    return ret

if __name__=="__main__":
    ret=evaluate_cityscapes("val10_dir","cityscapes_dataset/gtFine/val")
    print(ret)
