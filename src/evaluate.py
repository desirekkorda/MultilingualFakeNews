
import torch

def eval_epoch(
    model,
    data_loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in data_loader:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch["label"].to(device)

            outputs = model(
                input_ids,
                attention_mask
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            preds = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return (
        total_loss / len(data_loader),
        accuracy
    )
