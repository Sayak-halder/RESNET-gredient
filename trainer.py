import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from analysis import GradientAnalyzer

def train_and_analyze(model, model_name, train_loader):
    model = model.to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM)
    
    analyzer = GradientAnalyzer()
    analyzer.register_hooks(model)
    
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    gradient_history = {name: [] for name in layer_names}
    
    print(f"\n🚀 Training {model_name}...")
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_grads = {name: 0 for name in layer_names}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for images, labels in progress_bar:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            batch_grads = analyzer.get_norms()
            for name in layer_names:
                if name in batch_grads:
                    epoch_grads[name] += batch_grads[name]
                    
            progress_bar.set_postfix({"loss": loss.item()})
            
        num_batches = len(train_loader)
        for name in layer_names:
            gradient_history[name].append(epoch_grads[name] / num_batches)
            
    analyzer.remove_hooks()
    return gradient_history