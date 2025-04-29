import torch
class AvgMeter:
    def __init__(self, accuracies_req_num=7):
        self.name = "Metric"
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.lr = 0.0
        self.mr1_to_mr2_accuracy = torch.zeros(accuracies_req_num)
        self.mr2_to_mr1_accuracy = torch.zeros(accuracies_req_num)
        self.mr1_to_mr2_accuracy_sum = torch.zeros(accuracies_req_num)
        self.mr2_to_mr1_accuracy_sum = torch.zeros(accuracies_req_num)
        self.element12_accuracy = 0.0
        self.element12_accuracy_sum = 0.0
        self.element21_accuracy = 0.0
        self.element21_accuracy_sum = 0.0

        self.loss_mr_avg = 0.0
        self.loss_mr_sum = 0.0
        self.loss_element_avg = 0.0
        self.loss_element_sum = 0.0

    def update(self, loss, loss_mr, loss_element, mr1_2_accuracy, mr2_1_accuracy, element12_accuracy, element21_accuracy):
        self.count += 1

        self.sum += loss
        self.avg = self.sum / self.count

        self.loss_mr_sum += loss_mr
        self.loss_mr_avg = self.loss_mr_sum/self.count
 
        self.loss_element_sum += loss_element
        self.loss_element_avg = self.loss_element_sum/self.count

        self.mr1_to_mr2_accuracy_sum += mr1_2_accuracy
        self.mr2_to_mr1_accuracy_sum += mr2_1_accuracy
        self.element12_accuracy_sum += element12_accuracy
        self.element21_accuracy_sum += element21_accuracy

        self.mr1_to_mr2_accuracy = self.mr1_to_mr2_accuracy_sum/self.count
        self.mr2_to_mr1_accuracy = self.mr2_to_mr1_accuracy_sum/self.count
        self.element12_accuracy = self.element12_accuracy_sum/self.count
        self.element21_accuracy = self.element21_accuracy_sum/self.count

    def get_lr(self, optimizer):
        self.lr = optimizer.param_groups[0]['lr']
        return self.lr

    def __repr__(self):
        text = f"{self.name}: {self.avg:.8f}, MR1 to MR2 Accuracy: {self.mr1_to_mr2_accuracy}, MR2 to MR1 Accuracy: {self.mr2_to_mr1_accuracy}"
        return text
