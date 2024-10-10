import torch
import numpy as np
import os


def calculate_iou_loss(pred_masks, target_masks, pred_score):
    """
    Calculate intersection over union (IoU) between predicted and target masks.
    Parameters:
        pred_masks (torch.Tensor): predicted masks
        target_masks (torch.Tensor): target masks
    Returns:
        float: value of IoU
    """
    pred = pred_masks.detach().cpu().numpy()
    target = np.any(target_masks.detach().cpu().numpy(),axis = 0)
    size = len([scr for scr in pred_score if scr>0.9])
    new_pred = np.zeros([size,np.shape(pred)[-1],np.shape(pred)[-1]])
    for index in range(len(pred_masks)):
        if pred_score[index] > 0.9:
            new_pred[index] = pred[index]
    pred = new_pred
    intersection = np.sum(np.logical_and(pred,target))
    union = np.sum(pred) + np.sum(target)
    iou = (2*intersection) / (union)
    
    
    return iou

def train_model(model, optimizer, data_train, data_valid, device, out_path, batch_size, lr):
    train_loss , valid_loss, valid_IoU  = [] , [], []
    train_m_loss , valid_m_loss         = [] , []
    f = open(os.path.join(out_path, "RCNN_log.txt"), "a")
    for i in range(101): #Epochs
        for phase in ['train','valid']:
            running_loss = 0
            running_m_loss = 0
            total_iou_mask = 0
            
            if phase == 'train':
                model.train(True)
                datal = data_train
            else:
                model.train(False)
                datal = data_valid
            
            for cycle in range(int(np.ceil(len(datal)/batch_size))):
                
                images, targets= datal.get_data(batch_size)
                images = list(image.to(device) for image in images)
                targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    # losses = loss_dict['loss_mask']
                    running_loss+= loss_dict['loss_mask'] * batch_size
                    losses.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        model.eval()
                        outputs = model(images,targets)
                        model.train(True)
                        loss_valid = model(images, targets)
                        model.train(False)
                        losses = sum(loss for loss in loss_valid.values())
                        running_loss+= losses * batch_size
                        running_m_loss += loss_valid['loss_mask'] * batch_size
                        for j in range(len(outputs)):
                            preds_masks = outputs[j]['masks'] > 0.5  # Convert mask probabilities to binary masks
                            preds_score = outputs[j]['scores']
                            gt_masks = targets[j]['masks']
                            iou_mask = calculate_iou_loss(preds_masks, gt_masks, preds_score)
                            total_iou_mask += iou_mask
            epoch_loss = running_loss/len(datal)
            epoch_m_loss = running_m_loss/len(datal)
            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)
            train_m_loss.append(epoch_m_loss) if phase == 'train' else valid_m_loss.append(epoch_m_loss)
        avg_iou_mask = total_iou_mask/ len(datal)
        valid_IoU.append(avg_iou_mask)
        f.write('Epoch: {} ; Train_loss: {} ; Valid_loss: {} ; Valid_dice: {}\n'.format(i, train_loss[-1],valid_loss[-1],valid_IoU[-1]))
        # Save every 10 epochs as 'temp_save.pth'
        if i % 10 == 0:
            save_checkpoint(model, optimizer, i, out_path, train_loss, valid_loss, valid_IoU, train_m_loss, valid_m_loss, batch_size, lr, temp=True)
    
    # Save the final model at the end of all epochs as 'normal.pth'
    save_checkpoint(model, optimizer, i, out_path, train_loss, valid_loss, valid_IoU, train_m_loss, valid_m_loss, batch_size, lr, temp=False)
    
    f.close()


   
def save_checkpoint(model, optimizer, i, out_path, train_loss, valid_loss, valid_IoU, train_m_loss, valid_m_loss, batch_size, lr, temp=True):
    if temp:
        # Save as 'temp_save.pth' every 10 epochs
        torch.save({
            'model_state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
            'loss': 'CrossEntropy',
            'epoch': i,
            'bs': batch_size, 
            'lr': lr,
            'valid_loss': valid_loss,
            'valid_dice': valid_IoU,
            'train_loss': train_loss,
            'tmask_loss': train_m_loss,
            'vmask_loss': valid_m_loss,
        }, out_path + "/temp_save.pth")
    else:
        # Save the final model as 'normal.pth' at the end
        torch.save({
            'model_state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
            'loss': 'CrossEntropy',
            'epoch': i,
            'bs': batch_size, 
            'lr': lr,
            'valid_loss': valid_loss,
            'valid_dice': valid_IoU,
            'train_loss': train_loss,
            'tmask_loss': train_m_loss,
            'vmask_loss': valid_m_loss,
        }, out_path + "/normal.pth")
