"""ADD-RSC adaptive differential denoising and bias-denoising objective."""
from torch import nn

from ..encoders import image_shape
from ..upstream import load_reference
from .base import StepResult, TrainingMethod


class DenoisedEncoder(nn.Module):
    def __init__(self, encoder, n_freq, args):
        super().__init__()
        self.encoder, self.final_feat_dim, self.mlp_head = encoder, encoder.final_feat_dim, encoder.mlp_head
        layer = load_reference("ADD-RSC", "models/adapt_diff_denoise.py").DiffTransformerLayer
        self.denoise = layer(d_model=n_freq, num_heads=args.denoise_heads, depth=args.denoise_depth)
        self.denoise.classifier = nn.Linear(n_freq, args.n_cls)
        self.aux = None

    def forward(self, x, **kwargs):
        z, self.aux = self.denoise(x[:, 0])
        return self.encoder(z.unsqueeze(1), **kwargs)


class Method(TrainingMethod):
    trainable_frontend = True
    supports_data_parallel = False  # reference denoiser stores auxiliary logits on its encoder

    def wrap_encoder(self, model, spec, args):
        return DenoisedEncoder(model, image_shape(spec, args)[1], args)

    def encoder_parameters(self, model):
        return model.encoder.parameters()

    def losses(self, model, args):
        return [*super().losses(model, args),
                load_reference("ADD-RSC", "models/bias_denoise_loss.py").LabelSmoothingLoss(size=args.n_cls)]

    def step(self, context, batch):
        c, a = context, context.args
        output = c.classifier(c.encode(batch.images))
        loss = (1 - a.loss_beta) * c.criteria[0](output, batch.labels) + a.loss_beta * c.criteria[1](c.model.aux, batch.labels)
        return StepResult(output, loss, batch.labels)
