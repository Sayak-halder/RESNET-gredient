import matplotlib.pyplot as plt
import os
from config import Config

class GradientAnalyzer:
    """Captures gradient norms to prove skip connections prevent vanishing gradients."""
    def __init__(self):
        self.gradients = {}
        self.hooks = []

    def register_hooks(self, model):
        for name, layer in model.named_children():
            if name.startswith('layer'):
                hook = layer.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)

    def _get_hook(self, name):
        def hook(module, input, output):
            output.register_hook(lambda grad: self.gradients.update({name: grad.detach().norm().item()}))
        return hook

    def get_norms(self):
        norms = self.gradients.copy()
        self.gradients.clear()
        return norms

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def plot_gradient_flow(gradient_histories):
    """The Twitter Hook: Visualizing the vanishing gradient problem."""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    colors = {'Standard ResNet': 'blue', 'SE-ResNet': 'green'}
    
    for model_name, history in gradient_histories.items():
        final_epoch_grads = [history[layer][-1] for layer in ['layer1', 'layer2', 'layer3', 'layer4']]
        plt.plot(['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'], final_epoch_grads, 
                 marker='o', label=model_name, color=colors.get(model_name, 'black'), linewidth=2)
        
    plt.title('Gradient Flow Across ResNet Layers (Final Epoch)', fontsize=14, fontweight='bold')
    plt.ylabel('Gradient Norm (Log Scale)', fontsize=12)
    plt.xlabel('Network Depth', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(Config.OUTPUT_DIR, 'gradient_flow_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ Gradient flow chart saved as '{save_path}'")