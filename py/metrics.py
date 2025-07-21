from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def calculate_precision(pred, target):
    return precision_score(target, pred, average='binary', pos_label=1, zero_division=0)
    
def calculate_recall(pred, target):
    return recall_score(target, pred, average='binary', pos_label=1, zero_division=0)

def calculate_f1(pred, target):
    return f1_score(target, pred, average='binary', pos_label=1, zero_division=0)

def calculate_jaccard_building(pred, target): #IoU solo per la classe building
    return jaccard_score(target, pred, average='binary', pos_label=1, zero_division=0)

def calculate_mean_iou(pred, target): # mIoU generale (più utile nel caso multiclasse)
    return jaccard_score(target, pred, average='macro', zero_division=0)

def calculate_pixel_accuracy(pred, target):
    return (target == pred).mean()

def calculate_dice_coefficient(pred, target, offset=1e-8):
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + offset) / (pred.sum() + target.sum() + offset)
    return dice