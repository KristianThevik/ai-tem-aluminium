import os
import torch
import time

def dice(predb, yb, device):
    pb = predb.argmax(dim=1)
    yb = yb.to(device)
    
    cls_ = predb.shape[1]
    
    mean_dice = 0
    dice_list = torch.zeros(cls_)
    
    for i in range(cls_):
        p = pb == i
        y = yb == i
    
        volume_sum = torch.sum(y, dim=(2,1)) + torch.sum(p, dim=(2,1))

        volume_intersect = torch.logical_and(p, y)
        volume_intersect = torch.sum(volume_intersect, dim=(2,1))
        
        # if volume_sum
        dice = (2*volume_intersect / volume_sum).mean()
        
        mean_dice += dice
        dice_list[i] = dice
 
    return mean_dice/cls_, (dice_list)


def save_model(model, optimizer, train_loss, valid_loss, valid_met, epoch, datal, out_path, learning_rate, early=False):
    if early:
        f_path = os.path.join(out_path, 'Unet_patience.pth')
        print(f"Early save at epoch {epoch}")
    else:
        f_path = out_path + 'Unet_model.pth'

    torch.save({
        'model_state_dict': model.state_dict(),
        'optim': optimizer.state_dict(),
        'loss': 'CrossEntropy',
        'epoch': epoch,
        'batch_size': datal.batch_size,
        'lr' : learning_rate, 
        'train_loss': train_loss, 
        'valid_loss': valid_loss,
        'valid_met': valid_met,
    }, f_path)
    print(f"Checkpoint saved at {f_path}")

def train_model(model, optimizer, loss_function, metric, train_data, valid_data, number_of_epochs, patience, out_path, learning_rate, device):
    train_loss, valid_loss = [], []
    train_met, valid_met = [], []
    
    save = True
    start = time.time()

    for epoch in range(number_of_epochs):
        f = open(os.path.join(out_path,'Unet_log.txt'), "a")
        f.write('-' * 100 + '\n')
        f.write('Epoch {}/{}\n'.format(epoch, number_of_epochs - 1))

        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                model.train(True)
                datal = train_data
            else:
                model.train(False)
                datal = valid_data

            running_loss = 0.0
            running_met = 0.0
            running_met_arr = torch.zeros(2)

            step = 0
            for x, y in datal:
                x = x.to(device)
                y = y.to(device)
                step += 1

                if phase == 'Train':
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_function(outputs, y)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_function(outputs, y)

                with torch.no_grad():
                    met, met_list = metric(outputs, y, device)

                running_met += met * datal.batch_size
                running_loss += loss * datal.batch_size
                running_met_arr += met_list * datal.batch_size

            epoch_loss = running_loss / len(datal.dataset)
            epoch_met = running_met / len(datal.dataset)
            epoch_met_arr = running_met_arr / len(datal.dataset)

            f.write('{} Loss: {:.4f}; Metric: {:.4f}\n'.format(phase, epoch_loss, epoch_met))
            print('Dice #1: {:.4f}, Dice #2: {:.4f}'.format(epoch_met_arr[0], epoch_met_arr[1]))

            if phase == 'Train':
                train_loss.append(epoch_loss)
                train_met.append(epoch_met)
            else:
                valid_loss.append(epoch_loss)
                valid_met.append(epoch_met)

        if (len(valid_loss) > patience) and (len(valid_loss) - torch.argmin(torch.tensor(valid_loss)) >= patience) and save:
            save = False
            f.write(f"Early stopping at epoch: {epoch}, patience: {patience}\n")
            save_model(model, optimizer, train_loss, valid_loss, valid_met, epoch, datal, out_path, learning_rate, early=True)
            f.close()
            break

        f.close()

    save_model(model, optimizer, train_loss, valid_loss, valid_met, number_of_epochs, datal, out_path, learning_rate)

    time_elapsed = time.time() - start
    f = open(os.path.join(out_path, 'Unet_log.txt'), "a")
    f.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    f.close()
    
