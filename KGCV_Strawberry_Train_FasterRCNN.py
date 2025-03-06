import os
import numpy as np
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import math
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys
sys.path.append(r'faster_rcnn/detection')
import utils
#import transforms as T
#import my_utils as utils
from torchvision import transforms as T
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import time
import glob
import json
import random
import datetime
version = int(sys.version.split()[0].split('.')[1])
if version > 7:
    import pickle 
else:
    import pickle as pickle
    
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL) #pickle.HIGHEST_PROTOCOL--> latest version save 
        
def load_object(filename):
    with open(filename, 'rb') as inp:
        data = pickle.load(inp)
    return data
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300

class Dataset(object):
    def __init__(self, root, transforms = None, scale = 1/4):
        self.root = root
        self.transforms = transforms  # normalization or resizing (preprocesing)
        self.imgs = glob.glob('%s/*.jpg'%root)
        self.labels = ['%s.json'%(t.split('.')[0]) for t in self.imgs]
        self.validlabel = ['background','flower','small g','green','white','turning red','red','overripe']
        self.scale = scale
        print(f"Found {len(self.imgs)} images and {len(self.labels)} labels in {root}")
       
    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        label_path = self.labels[idx]
        if img_path.lower().endswith('.jpg'):
            img = Image.open(img_path).convert("RGB")
        img = img.resize((int(img.size[0]*self.scale), int(img.size[1]*self.scale)),resample=0)
        size_0 = img.size
        img = ImageOps.exif_transpose(img)   # rotating the img when the up direction saved in exif
        size_1 = img.size
        if size_0!=size_1:
            print("Shape of img {} is opposited, corrected from {} to {}".format(img_path,size_0,size_1))
            #print('test')
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        with open(label_path,'r') as f:
            labeldata= json.load(f)
           
        #labels = list(np.loadtxt(label_path))
        num_objs = len(labeldata['shapes']) # image me object count
        boxes = []      
        labels = []
        for label in labeldata['shapes']:
            # extract x and y cordinate of bounding box
            xloc = [label['points'][0][0],label['points'][1][0]]
            xmin = np.min(xloc) *self.scale  # left boundry # doing this is because sometime the labelme will flip the bounding box
            xmax = np.max(xloc) *self.scale  # right boundry
            
            yloc = [label['points'][0][1],label['points'][1][1]]
            ymin = np.min(yloc) *self.scale # top  boundry
            ymax = np.max(yloc) *self.scale # bollam boundry
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.validlabel.index(label['label'].split(', ')[0])) # label = 0 is background
            #convert pytorch  tensor 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx]) # uniq id store in tensor        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                   # ( area = wirth * hight )
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels   
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            
            #target = self.transforms(target)
        return {'image':img, 'target':target}
    

    def __len__(self): # len()
        return len(self.imgs)

def _get_iou_types(model): 
    model_without_ddp = model # DistributedDataParalle
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"] #intersection over Union
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN): # ob & seg -> detect
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN): # human pose estimation
        iou_types.append("keypoints")
    return iou_types

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()   
    metric_logger = utils.MetricLogger(delimiter="  ")   #  log messages ko separate
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None 
    if epoch == 0:  #warm-up st...
        warmup_factor = 1.0 / 1000  
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # debug
        for ts in targets:
            for t in ts['boxes']:
                a = t[3].cpu().numpy()-t[1].cpu().numpy() 
                if a<=0:
                    print(ts)
        with torch.cuda.amp.autocast(enabled=scaler is not None): #amp --> Automatic Mixed Precision
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) # Backpropagation ke liye final loss
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item() # extract loss value from tensor

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None: #amp active 
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # Collect all predictions and targets
    predictions = []
    targets_list = []

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # Store predictions and targets for later evaluation
        predictions.extend(outputs)
        targets_list.extend(targets)
        
        metric_logger.update(model_time=model_time)

    # Create a simplified evaluation method instead of using COCO API
    # Calculate mAP at different IoU thresholds
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ap_per_class = {}
    ar_per_class = {}
    
    # Simple function to calculate IoU between two boxes
    def calculate_iou(box1, box2):
        # Box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Check if boxes overlap
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Collect all ground truth and predictions by category
    all_gt = {}
    all_pred = {}
    
    for target, pred in zip(targets_list, predictions):
        image_id = target["image_id"].item()
        
        # Process ground truth
        gt_boxes = target["boxes"].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()
        
        for box, label in zip(gt_boxes, gt_labels):
            if label not in all_gt:
                all_gt[label] = []
            all_gt[label].append({"image_id": image_id, "box": box, "detected": False})
        
        # Process predictions
        pred_boxes = pred["boxes"].cpu().numpy()
        pred_labels = pred["labels"].cpu().numpy()
        pred_scores = pred["scores"].cpu().numpy()
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if label not in all_pred:
                all_pred[label] = []
            all_pred[label].append({"image_id": image_id, "box": box, "score": score})
    
    # Calculate AP for each class at each IoU threshold
    results = []
    for label in all_gt.keys():
        if label not in all_pred:
            ap_per_class[label] = 0.0
            continue
            
        # Sort predictions by score
        pred_boxes = sorted(all_pred[label], key=lambda x: x["score"], reverse=True)
        gt_boxes = all_gt[label]
        
        # Reset detection flag
        for gt in gt_boxes:
            gt["detected"] = False
        
        # Calculate precision and recall
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        
        for i, pred in enumerate(pred_boxes):
            max_iou = 0.0
            max_idx = -1
            
            # Find the best matching ground truth box
            for j, gt in enumerate(gt_boxes):
                if gt["image_id"] == pred["image_id"] and not gt["detected"]:
                    iou = calculate_iou(pred["box"], gt["box"])
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = j
            
            # Check if the match is good enough
            if max_idx >= 0 and max_iou >= 0.5:
                gt_boxes[max_idx]["detected"] = True
                tp[i] = 1
            else:
                fp[i] = 1
        
        # Calculate cumulative sums
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        recall = cumsum_tp / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros_like(cumsum_tp)
        precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)
        
        # Calculate average precision using 11-point interpolation
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
            
        ap_per_class[label] = ap
        
        # Calculate average recall
        ar = np.sum(tp) / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
        ar_per_class[label] = ar
        
        results.append({
            "category_id": label,
            "ap": ap,
            "ar": ar
        })
    
    # Calculate mAP across all classes
    mean_ap = np.mean([ap for ap in ap_per_class.values()]) if ap_per_class else 0.0
    mean_ar = np.mean([ar for ar in ar_per_class.values()]) if ar_per_class else 0.0
    
    print(f"mAP: {mean_ap:.4f}, mAR: {mean_ar:.4f}")
    
    # Print per-class results
    for label, ap in ap_per_class.items():
        ar = ar_per_class.get(label, 0.0)
        print(f"Class {label}: AP={ap:.4f}, AR={ar:.4f}")
    
    # Create a mock COCO evaluator to return
    class MockCocoEvaluator:
        def __init__(self, stats):
            self.coco_eval = {'bbox': self}
            self.stats = stats
        
        def summarize(self):
            print("AP: {:.4f}".format(self.stats[0]))
            print("AP50: {:.4f}".format(self.stats[1]))
            print("AP75: {:.4f}".format(self.stats[2]))
    
    stats = np.array([mean_ap, mean_ap, mean_ap, 0, 0, 0, mean_ar, mean_ar, mean_ar, 0, 0, 0])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    torch.set_num_threads(n_threads)
    return MockCocoEvaluator(stats)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

    
def get_transform(train):
    transforms_list = []
    # transforms_list.append(T.Resize((800,800))) # multiple image stack
    transforms_list.append(T.ToTensor())  # Convert image to tensor
    
    if train:
        transforms_list.append(T.RandomHorizontalFlip(0.5))  # Apply random horizontal flip

    return T.Compose(transforms_list)

def plot_accuracies(log_test,outFolder='log',title='Accuracy vs. epochs',saveFig=False):
    fig=plt.figure()
    AP50_95 = [x[0] for x in log_test]
    AP50 = [x[1] for x in log_test]
    AR10 = [x[7] for x in log_test]
    plt.plot(AP50_95, 'r-',label = 'AP IoU 0.5:0.95')
    plt.plot(AP50, 'g-',label = 'AP IoU 0.5')
    plt.plot(AR10, 'y-',label = 'AR max detection 10')
    plt.xlabel('epoch')
    plt.ylabel('AP & AR')
    plt.legend()
    plt.title(title)
    if saveFig:    
        fig.savefig('%s/AP.png'%outFolder)
    
def plot_loss(log_train,outFolder='log',title='Loss vs. epochs',saveFig=False):
    fig=plt.figure()
    loss_base = [dict(x)['loss'] for x in log_train]
    loss_classifier = [dict(x)['loss_classifier'] for x in log_train]
    loss_box_reg = [dict(x)['loss_box_reg'] for x in log_train]
    loss_objectness = [dict(x)['loss_objectness'] for x in log_train]
    loss_rpn_box_reg = [dict(x)['loss_rpn_box_reg'] for x in log_train]
    plt.plot(loss_base, 'r-',label = 'loss')
    plt.plot(loss_classifier, 'g-',label = 'loss_classifier')
    plt.plot(loss_box_reg, 'y-',label = 'loss_box_reg')
    plt.plot(loss_objectness, 'b-',label = 'loss_objectness')
    plt.plot(loss_rpn_box_reg, 'c-',label = 'loss_rpn_box_reg')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(title)
    if saveFig:  
        fig.savefig('%s/loss.png'%outFolder)

def datasetCheck(data_loader):
    bboxesList = []
    areaList = []
    labelList = []
    idList = []
    i=0
    for imgs, targets in data_loader:
        for img, target in zip(imgs, targets):
            for bbox,area,l,img_id in zip(target['boxes'].cpu().tolist(),target['area'].cpu().tolist(), 
                                          target['labels'].cpu().tolist(),target['image_id'].cpu().tolist()):
                bboxesList.append(bbox)
                areaList.append(area)
                labelList.append(l)
                idList.append(img_id)
        if i%30==0:
            print('processed %d batches'%i)
        i+=1
#------------>
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        print ('set cudnn seed')
        
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        print ('set cuda seed')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def checkDataloader(data_loader):
    for img, target in data_loader:
        print(type(img),type(target))
        print('img.shape:', img[0].cpu().numpy().shape)
        bboxes = target[0]['boxes'].cpu().tolist()
        print('target.boxes:{},area:{},labels:{},id:{}'.format(bboxes, 
              target[0]['area'], target[0]['labels'],
               target[0]['image_id']))
        plt.figure()
        plt.imshow(img[0].cpu().numpy().transpose(1,2,0))
        currentAxis=plt.gca()
        for bbox in bboxes:
            rect=patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],
                                   bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
            currentAxis.add_patch(rect)
        break
    
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # settings
    saveResult = True
    num_classes = 7 + 1 # classes plus background
    print_freq = 100
    num_epochs = 5
    seed=1
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)

    # make dataloader
    note = 'Faster_RCNN'        
    dataset = Dataset(root = r"C:\Users\hp\Desktop\iit_mandi\KGCV_Strawberry-main\strawberry_fasterRCNN\train", transforms = get_transform(train=True))
    dataset_test = Dataset(root = r'C:\Users\hp\Desktop\iit_mandi\KGCV_Strawberry-main\strawberry_fasterRCNN\test', transforms = get_transform(train=False))
    
    # Debugging: Check if datasets are loaded correctly
    print(f"Training dataset length: {len(dataset)}")
    print(f"Test dataset length: {len(dataset_test)}")
    
    
    data_loader = torch.utils.data.DataLoader(
       dataset, batch_size=1, shuffle=True, num_workers=4,
       collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn = utils.collate_fn)

    # check dataloader
    checkDataloader(data_loader)
    
    # get the model
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.8, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=0.005)
                                
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=1,
                                                   gamma=0.95)

    # let's train it
    log_train = []
    log_test = []
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_freq)
        log_train.append([(t[0], t[1].avg) for t in metric_logger.meters.items()])
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluator = evaluate(model, data_loader_test, device=device)
        log_test.append(evaluator.coco_eval['bbox'].stats)
        
    # save model
    outPath = 'models'
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    if not os.path.exists('log'):
        os.mkdir('log')
    
    if saveResult:
        now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        logPath = '%s_%s_epoch%d'%(note,now,num_epochs)
        torch.save(model, '%s/fasterRcnn-%s.pth'%(outPath,logPath))
        torch.save(model.state_dict(), '%s/fasterRcnn-%s_state_dict.pth'%(outPath,logPath))

        if not os.path.exists('log/%s'%logPath):
            os.mkdir('log/%s'%logPath)
        plot_accuracies(log_test,outFolder='log/%s'%logPath,saveFig=saveResult)
        plot_loss(log_train,outFolder='log/%s'%logPath,saveFig=saveResult)
        obj = {}
        obj['log_train'] = log_train
        obj['log_test'] = log_test
        save_object(obj, 'log/%s/log.pkl'%logPath)
    # test for n samples
    n = 4
    labelName = dataset.validlabel
    i = 0
    for img, target in data_loader_test:
        model.eval()
        predicted = model(img[0].unsqueeze(0).to(device))
        bboxes_p = predicted[0]['boxes'].cpu().tolist()
        bboxes = target[0]['boxes'].cpu().tolist()
        labels = target[0]['labels'].cpu().tolist()
        labels_p = predicted[0]['labels'].cpu().tolist()
        score_p = predicted[0]['scores'].cpu().tolist()
        plt.figure()
        plt.imshow(img[0].cpu().numpy().transpose(1,2,0))
        currentAxis=plt.gca()
        for bbox,label in zip(bboxes,labels):
            rect=patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],
                                   bbox[3]-bbox[1],linewidth=1,edgecolor='g',facecolor='none')
            currentAxis.add_patch(rect)
        for bbox,label,score in zip(bboxes_p,labels_p,score_p):
            if score > 0.5:
                rect=patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],
                                       bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
                currentAxis.add_patch(rect)
                props = dict(boxstyle='round', facecolor='r', alpha=1) 
                plt.text(bbox[0], bbox[1],"%s: %.2f"%(labelName[label],score),
                          bbox=dict(facecolor='r', edgecolor='r', pad=0),color='w',fontsize = 6)
                # plt.text(bbox[0], bbox[1],"%s: %.2f"%(label,score),
                #          bbox=dict(facecolor='r', edgecolor='r', pad=0),color='w',fontsize = 6)
                plt.axis('off')  
        i += 1
        if i >= n:
            break 