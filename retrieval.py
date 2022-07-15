from model import *
from config import *

def prediction(loader):
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)  
        if i == 0:
            outputs = feature_model(images)
            label = labels
        else:
            outputs = torch.cat((outputs, feature_model(images)), 0)
            label = torch.cat((label, labels), 0)
    if train_flag and save_flag:
        f.write('output_sample: \n' + str(outputs[:10]) + '\n')
        f.write('label_sample: \n' + str(label[:10]) + '\n')
 
    return outputs.cpu().numpy(), label.cpu().numpy()

def cos_dist(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    return 1 - np.dot(a,b)/(a_norm * b_norm)

# Test the model
def test():
    feature_model.eval()
    if train_flag or not os.path.exists(path + '.data'):
        with torch.no_grad():
            data_predict, data_label = prediction(databaseloader)
            test_predict, test_label = prediction(testloader)
        
        if not train_flag and save_flag:
            datafile = open(path + '.data', 'w')
            datafile.write(json.dumps([data_predict.tolist(), data_label.tolist(), test_predict.tolist(), test_label.tolist()]))
            datafile.close()
            print('------------- save data -------------')
    else:
        datafile = open(path + '.data', 'r').read()
        data = json.loads(datafile)
        data_predict = np.array(data[0])
        data_label = np.array(data[1])
        test_predict = np.array(data[2])
        test_label = np.array(data[3])
        print('------------- load data -------------')

    data_predict = np.sign(data_predict)
    test_predict = np.sign(test_predict)
    similarity = 1 - np.dot(test_predict, data_predict.T) / num_bits
    sim_ord = np.argsort(similarity, axis=1)

    apall=np.zeros(test_num)
    for i in range(test_num):
        x=0
        p=0
        order=sim_ord[i]
        for j in range(retrieve):
            if np.dot(test_label[i], data_label[order[j]]) > 0:
                x += 1
                p += float(x) / (j + 1)
        if p > 0:   
            apall[i] = p / x
    mAP=np.mean(apall)
    return mAP
  
if backbone == 'googlenet':
    feature_model = torchvision.models.inception_v3(pretrained = True)
    inchannel = feature_model.fc.in_features
    feature_model.fc = nn.Linear(inchannel, num_bits)
elif backbone == 'resnet':
    feature_model = torchvision.models.resnet50(pretrained = True)
    inchannel = feature_model.fc.in_features
    feature_model.fc = nn.Linear(inchannel, num_bits)
elif backbone == 'alexnet':
    feature_model = AlexNet(hash_bit = num_bits)
feature_model.to(device)

if train_flag:    # Train the model
    model = HyP().to(device)
    optimizer = torch.optim.SGD([{'params': feature_model.parameters(), 'lr':feature_rate}, {'params': model.parameters(), 'lr':criterion_rate}], momentum = 0.9, weight_decay = 0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.5)
    print('------------- model initialized -------------')

    total_step = len(trainloader)
    best_map = 0    #best map
    best_epoch = 0  #best epoch
    total_time = 0
    mAPs = []
    Times = []
    for epoch in range(num_epochs):
        feature_model.train()
        for i, (images, labels) in enumerate(trainloader):
            start = time.time()

            batch_x = images.to(device)
            batch_y = labels.to(device)  
            
            if backbone == 'googlenet': 
                hash_value = feature_model(batch_x)[0]
            else:
                hash_value = feature_model(batch_x)
            loss = model(x = hash_value, batch_y = batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()                      
            total_time += time.time() - start
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Time: %.4f'
                %(epoch , num_epochs, i, total_step, loss.item(), total_time))
            # print(proxies)
            if save_flag:
                f.write('| Epoch [' + str(epoch) + '/' + str(num_epochs) + '] Iter[' + str(i) + '/' + str(total_step) + '] Loss:' + str(loss.item()) + '\n')

        scheduler.step()

        if epoch % 5 == 4:
            mAP = test()
            if mAP > best_map:
                best_map = mAP
                best_epoch = epoch
                print("epoch: ", epoch)
                print("best_map: ", best_map)
                if save_flag:
                    f.write("epoch: " + str(epoch) + '\n')
                    f.write("best_map: " + str(best_map) + '\n')
                    torch.save(feature_model.state_dict(), model_path)
            else:
                print("epoch: ", epoch)
                print("map: ", mAP)
                print("best_epoch: ", best_epoch)
                print("best_map: ", best_map)
                if save_flag:
                    f.write("epoch: " + str(epoch) + '\n')
                    f.write("map: " + str(mAP) + '\n')
                    f.write("best_epoch: " + str(best_epoch) + '\n')
                    f.write("best_map: " + str(best_map) + '\n')

    if save_flag:
        f.write("best_map: " + str(best_map) + '\n')
        f.close()

else:
    if not os.path.exists(path + '.data'):
        feature_model.load_state_dict(torch.load(model_path,  map_location = device))
    best_map = test()

print("map: ", best_map)
    