import torch
import matplotlib.pyplot as plt

def calculate_accuracy(model, data_loader):
    """Calcule la précision globale du modèle."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            # Pour le RNN (Partie 3) ou MLP/CNN (Partie 1 & 2)
            if outputs.shape[1] == 1: # Cas binaire (Sigmoid)
                preds = (outputs > 0.5).float()
            else: # Cas multi-classes (Softmax)
                _, preds = torch.max(outputs, 1)
                
            total += labels.size(0)
            correct += (preds.squeeze() == labels).sum().item()
    return correct / total

def plot_loss(loss_history, title="Évolution de la Perte"):
    """Affiche un graphique de la perte pendant l'entraînement."""
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel("Époque")
    plt.ylabel("Loss")
    plt.show()