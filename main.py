from config import Config
from data import get_dataloaders
from models import ResNetEngine, Bottleneck, SEBottleneck
from trainer import train_and_analyze
from analysis import plot_gradient_flow

def main():
    print("🧠 Initializing ResNet Anatomy Lab...")
    train_loader, _ = get_dataloaders()
    
    # Standard ResNet50 configuration: [3, 4, 6, 3]
    models_to_test = {
        "Standard ResNet": ResNetEngine(Bottleneck, [3, 4, 6, 3], num_classes=Config.NUM_CLASSES),
        "SE-ResNet": ResNetEngine(SEBottleneck, [3, 4, 6, 3], num_classes=Config.NUM_CLASSES)
    }
    
    all_histories = {}
    for name, model in models_to_test.items():
        history = train_and_analyze(model, name, train_loader)
        all_histories[name] = history
        
    plot_gradient_flow(all_histories)
    print("\n Analysis Complete! Check outputs/charts/")

if __name__ == "__main__":
    main()