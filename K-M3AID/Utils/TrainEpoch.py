from Utils.AvgMeter import AvgMeter
from tqdm import tqdm
from Utils.mr2mr import *
def train_epoch(model, train_loader, optimizer, lr_scheduler, step, accuracies_req):
    loss_meter = AvgMeter(len(accuracies_req))
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        loss, loss_mr, loss_element, RS_logits, IE_logits, IE_list, IEs = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        # do not allow gradient in calculating accuracy.
        mr1_mr2_acc = mr2mr_match(RS_logits.detach(), accuracies_req)
        mr2_mr1_acc = mr2mr_match(RS_logits.detach().T, accuracies_req)
        
        element12_acc = element_match(IE_logits.detach(), IEs.detach(), IE_list.detach())
        element21_acc = element_match(IE_logits.detach().T, IEs.detach().T, IE_list.detach())

        loss_meter.update(loss.item(), loss_mr.item(), loss_element.item(), mr1_mr2_acc, mr2_mr1_acc, element12_acc, element21_acc)
        loss_meter.get_lr(optimizer)

        tqdm_object.set_postfix(
            train_loss=loss_meter.avg,
            mr_loss = loss_meter.loss_mr_avg,
            element_loss = loss_meter.loss_element_avg,
            element12_acc = loss_meter.element12_accuracy,
            element21_acc = loss_meter.element21_accuracy,   
            mr1_to_mr2_accuracy=loss_meter.mr1_to_mr2_accuracy,
            mr2_to_mr1_accuracy=loss_meter.mr2_to_mr1_accuracy,
            lr=loss_meter.lr
        )
        #loss_meter.print_epoch_results() 

    return loss_meter
