import torch.nn as nn
from fairseq import utils
from fairseq.models.fairseq_model import BaseFairseqModel, check_type
from fairseq.models import FairseqEncoder
from fairseq.models.transformer import TransformerEncoder, Embedding

from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models import register_model, register_model_architecture

# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.


@register_model('dual_encoder_transformer')
class DualEncoderModel(BaseFairseqModel):

    def __init__(self, encoder1, encoder2, args, task):
        super().__init__()
        self.args = args

        self.encoder_src = encoder1
        self.encoder_tgt = encoder1 if encoder2 is None else encoder1
        self.bag_of_word = args.bag_of_word
        self.encoder_hidden_dim = args.encoder_embed_dim
        self.tgt_voc_size = len(task.source_dictionary)
        self.src_voc_size = len(task.target_dictionary)
        self.pad_tgt = task.target_dictionary.pad()

        self.Wsrc = nn.Linear(self.encoder_hidden_dim, self.src_voc_size, bias=False) if self.bag_of_word else None
        self.Wtgt = nn.Linear(self.encoder_hidden_dim, self.tgt_voc_size, bias=False) if self.bag_of_word else None

        check_type(self.encoder_src, FairseqEncoder)
        check_type(self.encoder_tgt, FairseqEncoder)

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        parser.add_argument(
            '--shared-encoder', action="store_true",
            help='the encoder is shared',
        )
        parser.add_argument(
            '--bag-of-word', action="store_true",
            help='the encoder is shared',
        )
        parser.add_argument(
            '--dropout', type=float, default=0.1,
            help='value of dropout',
        )
        parser.add_argument(
            '--encoder-embed-dim', type=int, default=1024,
            help='embedding dimension',
        )
        parser.add_argument(
            '--encoder-hidden-dim', type=int, default=1024,
            help='hidden state dimension',
        )
        parser.add_argument(
            '--encoder-layers', type=int, default=6,
            help='number of layers',
        )
        parser.add_argument(
            '--encoder-attention-heads', type=int, default=8,
            help='embedding dimension',
        )
        parser.add_argument(
            '--encoder-normalize-before', action="store_true",
            help='normalize before',
        )
        parser.add_argument(
            '--encoder-learned-pos', action="store_true",
            help='learn positional encoding',
        )
        parser.add_argument(
            '--activation_fn', default="relu",
            help='activation function',
        )
        parser.add_argument(
            '--apply-bert-init', action="store_true",
            help='weight init',
        )
        parser.add_argument(
            '--adaptive-input', action="store_true",
            help='adaptive input',
        )
        parser.add_argument(
            '--no-token-positional-embeddings', action="store_true",
            help='no token positional embeddings',
        )
        parser.add_argument(
            '--encoder-layerdrop', type=float, default=0.0,
            help='layerdrop rate',
        )
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        


    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # Initialize our Encoder and Decoder.

        # print(args)
        encoder_embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.encoder_embed_dim, args.encoder_embed_path
        )
        encoder1 = TransformerEncoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_tokens=encoder_embed_tokens,
        )
        if args.shared_encoder:
            encoder_embed_tokens2 = cls.build_embedding(
                args, task.target_dictionary, args.encoder_embed_dim, args.encoder_embed_path
            )
            encoder2 = TransformerEncoder(
                args=args,
                dictionary=task.target_dictionary,
                embed_tokens=encoder_embed_tokens2,
            )
        else:
            encoder2 = None
        model = DualEncoderModel(encoder1, encoder2, args, task)

        # Print the model architecture.
        print(model)

        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def forward(self, src_tokens, src_lengths, tgt_tokens, tgt_lengths):
        encoder_src_out = self.encoder_src(src_tokens, src_lengths)
        encoder_tgt_out = self.encoder_tgt(tgt_tokens, tgt_lengths)
        # print("encoder src all", encoder_src_out["encoder_out"][0].shape)
        encoder_src_out = nn.functional.normalize(encoder_src_out["encoder_out"][0][0, :], p=2, dim=-1)
        encoder_tgt_out = nn.functional.normalize(encoder_tgt_out["encoder_out"][0][0, :], p=2, dim=-1)
        out =  {
            "encoder_src_out": encoder_src_out,
            "encoder_tgt_out": encoder_tgt_out,
        }
        # print("+++++++++++++++++++++++++++++++++\n", (encoder_src_out.detach()**2).sum(-1))
        # print("=================================\n", (encoder_src_out.detach()**2).sum(-1))
        if self.bag_of_word:
            
            # print("encoder_src out", encoder_src_out.shape)
            out["src_bow_logits"] = self.Wsrc(encoder_src_out)
            out["tgt_bow_logits"] = self.Wtgt(encoder_tgt_out)

        return out

@register_model_architecture(
    "dual_encoder_transformer", "dual_encoder_transformer_base"
)
def multi_levenshtein_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_hidden_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.3)
    args.shared_encoder = getattr(args, "shared_encoder", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)
    args.bag_of_word = getattr(args, "bag_of_word", True)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", False)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", False)
    # args.apply_bert_init = getattr(args, "apply_bert_init", False)

    ...
