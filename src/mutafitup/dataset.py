from datasets import Dataset


def create_dataset(tokenizer, seqs, labels, checkpoint: str):
    tokenized = tokenizer(seqs, max_length=1024, padding=True, truncation=True)
    dataset = Dataset.from_dict(tokenized)

    if ("esm" in checkpoint) or ("ProstT5" in checkpoint):
        # we need to cut of labels after 1022 positions for the data collator to add the correct padding (1022 + 2 special tokens)
        labels = [label[:1022] for label in labels]
    else:
        # we need to cut of labels after 1023 positions for the data collator to add the correct padding (1023 + 1 special tokens)
        labels = [label[:1023] for label in labels]

    dataset = dataset.add_column("labels", labels)  # type: ignore

    return dataset


def create_per_residue_dataset(tokenizer, seqs, labels, checkpoint: str):

    return create_dataset(tokenizer, seqs, labels, checkpoint)


def create_per_protein_dataset(tokenizer, seqs, labels, checkpoint: str):

    tokenized = tokenizer(seqs, max_length=1024, padding=True, truncation=True)

    dataset = Dataset.from_dict(tokenized)

    dataset = dataset.add_column("labels", labels)  # type: ignore

    return dataset
