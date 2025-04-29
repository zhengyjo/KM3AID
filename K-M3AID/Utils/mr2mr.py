import torch
import torch.nn.functional as F


def calculate_top_accuracy(predicted_labels, k):
    correct_count = 0
    total_count = len(predicted_labels)

    for i in range(total_count):
        prediction = predicted_labels[i]

        if i in prediction[:k]:
            correct_count += 1

    top_accuracy = correct_count / total_count * 100
    return top_accuracy


def mr2mr_match(scores_matrix, accuracies_req):
    top_indices = torch.argsort(scores_matrix, dim=1, descending=True)[:, :]
    predicted_labels = top_indices.tolist()

    top_accuracy_list = []
    for k in accuracies_req:
        accuracy = calculate_top_accuracy(predicted_labels, k)
        top_accuracy_list.append(accuracy)

    accuracy_tensor = torch.tensor(top_accuracy_list, device='cpu')
    return accuracy_tensor



def element_calc(score, diff):
    top_indices = torch.argmax(score, dim=1) 
    predicts = top_indices.tolist()

    count = 0 
    for i in range(len(predicts)):
        if diff[i][predicts[i]] == 0:
            count +=1
            
    acc = count/len(predicts)*100       
    return acc 

def element_match(scores, ppm, IE_list):
    diffs = torch.abs(ppm - ppm.T)

    IE_list = IE_list.tolist()
    accList = []

    b = 0
    for i in IE_list:
        e = b + i
        score = scores[b:e, b:e]
        score = F.softmax(score, dim=0)
        diff = diffs[b:e, b:e]
        acc = element_calc(score, diff)
        accList.append(acc)
        b = e  
     
    accuracy_tensor = torch.tensor(accList)
    average = torch.mean(accuracy_tensor)
    return average.item()





