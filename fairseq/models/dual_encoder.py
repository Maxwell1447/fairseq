import torch.nn as nn
from fairseq import utils
from fairseq.models.fairseq_model import FairseqModel, check_type
from fairseq.models import FairseqEncoder

from fairseq.models import FairseqEncoderDecoderModel
# from fairseq.models import register_model, register_model_architecture

# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.


# @register_model('dual_encoder')
class DualEncoderModel(FairseqModel):

    def __init__(self, encoder1, encoder2, args, task):
        super().__init__()

        self.encoder_src = encoder1
        self.encoder_tgt = encoder1 if encoder2 is None else encoder1
        self.bag_of_word = args.bag_of_word
        self.encoder_hidden_dim = args.encoder_hidden_dim
        self.tgt_voc_size = len(task.source_dictionary)
        self.src_voc_size = len(task.target_dictionary)

        self.Wsrc = nn.Linear(self.encoder_hidden_dim, self.src_voc_size, bias=False) if self.bag_of_word else None
        self.Wtgt = nn.Linear(self.encoder_hidden_dim, self.tgt_voc_size, bias=False) if self.bag_of_word else None

        check_type(self.encoder_src, FairseqEncoder)
        check_type(self.encoder_tgt, FairseqEncoder)

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        parser.add_argument(
            '--shared-encoder', action="store_true",
            help='the encoder is shared',
        )
        parser.add_argument(
            '--bag-of-word', action="store_true",
            help='the encoder is shared',
        )

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # Initialize our Encoder and Decoder.
        encoder1 = FairseqEncoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_dim=args.encoder_hidden_dim,
            dropout=args.encoder_dropout,
        )
        if args.shared_encoder:
            encoder2 = FairseqEncoder(
                args=args,
                dictionary=task.target_dictionary,
                embed_dim=args.encoder_embed_dim,
                hidden_dim=args.encoder_hidden_dim,
                dropout=args.encoder_dropout,
            )
        else:
            encoder2 = None
        model = DualEncoderModel(encoder1, encoder2, args, task)

        # Print the model architecture.
        print(model)

        return model

    def forward(self, src_tokens, src_lengths, tgt_tokens, tgt_lengths):
        encoder_src_out = self.encoder_src(src_tokens, src_lengths)
        encoder_tgt_out = self.encoder_tgt(tgt_tokens, tgt_lengths)
        encoder_src_out = nn.functional.normalize(encoder_src_out, p=2, dim=-1)
        encoder_tgt_out = nn.functional.normalize(encoder_tgt_out, p=2, dim=-1)
        out =  {
            "encoder_src_out": encoder_src_out,
            "encoder_tgt_out": encoder_tgt_out,
        }
        if self.bag_of_word:
            out["src_bow_logits"] = self.Wsrc(encoder_src_out)
            out["tgt_bow_logits"] = self.Wtgt(encoder_tgt_out)

        return out

# @register_model_architecture(
#     "dual_encoder_transformer", "dual_encoder_transformer_base"
# )
# def multi_levenshtein_base_architecture(args):
#     ...
