import torch
from src.transformers.models.bert.modeling_bert import *

class ExtendedTransformer(BertPreTrainedModel):
    def __init__(self, config, transformer, num_labels, freeze_backbone=False):
        super().__init__(config)
        self.transformer = transformer
        self.linear = torch.nn.Linear(self.config.hidden_size, num_labels, bias=False)
        self.post_init()
        self.freeze_backbone = freeze_backbone
    def forward(self, *args, **inputs):
        if self.freeze_backbone:
            with torch.no_grad():
                embeds = self.transformer.base_model.embeddings.word_embeddings(inputs["input_ids"])
        else:
            embeds = self.transformer.base_model.embeddings.word_embeddings(inputs["input_ids"])
        inputs.pop("input_ids", None)
        inputs["inputs_embeds"] = embeds
        backbone_outputs = self.transformer(**inputs)
        sent_emb = torch.mean(backbone_outputs.hidden_states[-1], dim=1)
        output = clsModel.linear(sent_emb)
        return output