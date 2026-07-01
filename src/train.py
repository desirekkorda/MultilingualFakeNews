
import torch

def train_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch in data_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids,
            attention_mask
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / len(data_loader),
        correct / total
    )
