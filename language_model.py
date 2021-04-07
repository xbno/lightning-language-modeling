from argparse import ArgumentParser
import pytorch_lightning as pl
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    ReformerConfig,
    ReformerForMaskedLM,
    ReformerForSequenceClassification,
    ReformerModel,
    RobertaConfig,
    RobertaForMaskedLM,
)

# config = RobertaConfig(
#     vocab_size=52_000,
#     max_position_embeddings=514,
#     num_attention_heads=12,
#     num_hidden_layers=6,
#     type_vocab_size=1,
# )
# model = RobertaForMaskedLM(config=config)


from transformers.optimization import AdamW
from data import LMDataModule
import pl_data
import torch.nn.functional as F
import torch


class LMModel(pl.LightningModule):
    def __init__(
        self, model_name_or_path, learning_rate, adam_beta1, adam_beta2, adam_epsilon
    ):
        super().__init__()

        self.save_hyperparameters()

        # config = AutoConfig.from_pretrained(model_name_or_path, return_dict=True)
        # self.model = AutoModelForMaskedLM.from_pretrained(
        #     model_name_or_path, config=config
        # )
        config = ReformerConfig(
            attention_head_size=64,
            attn_layers=["local", "lsh", "local", "lsh", "local", "lsh"],
            axial_norm_std=1.0,
            axial_pos_embds=True,
            axial_pos_shape=[16, 32],  # 512
            # axial_pos_shape=[64, 64], # 4096
            # max_position_embeddings=4096, # [64, 64]
            axial_pos_embds_dim=[64, 192],
            chunk_size_lm_head=0,
            eos_token_id=2,
            feed_forward_size=512,
            hash_seed=None,
            hidden_act="relu",
            hidden_dropout_prob=0.05,
            hidden_size=256,
            initializer_range=0.02,
            is_decoder=False,
            layer_norm_eps=1e-12,
            local_num_chunks_before=1,
            local_num_chunks_after=0,
            local_attention_probs_dropout_prob=0.05,
            local_attn_chunk_length=64,
            lsh_attn_chunk_length=64,
            lsh_attention_probs_dropout_prob=0.0,
            lsh_num_chunks_before=1,
            lsh_num_chunks_after=0,
            num_attention_heads=4,
            num_buckets=None,
            num_hashes=1,
            pad_token_id=0,
            vocab_size=30_000,
            tie_word_embeddings=False,
            use_cache=True,
        )
        self.model = ReformerForMaskedLM(config)

    def forward(self, x):
        # return self.model(x).logits
        return self.model(x, return_dict=True).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch, return_dict=True).loss
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch, return_dict=True).loss
        self.log("valid_loss", loss, on_step=True)  # , sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_epsilon,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=5e-5)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.999)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        return parser


class LMElectraModel(pl.LightningModule):
    def __init__(self, lr, model_name_or_path, adam_beta1, adam_beta2, adam_epsilon):
        super().__init__()

        self.save_hyperparameters()
        # self.pad_token_id = -100
        # self.mask_token_id = 103
        self.pad_token_id = 0
        self.mask_token_id = 4
        self.vocab_size = 30_000
        self.temperature = 1.0
        self.gen_weight = 1.0
        self.disc_weight = 50.0
        self.lr = lr
        self.optimizers = []
        self.schedulers = []
        # self.tokenizer = tokenizer

        gen_config = ReformerConfig(
            attention_head_size=32,
            attn_layers=["local", "lsh", "local", "lsh", "local", "lsh"],
            axial_norm_std=1.0,
            axial_pos_embds=True,
            axial_pos_shape=[16, 32],  # 512
            # axial_pos_shape=[64, 64], # 4096
            # max_position_embeddings=4096, # [64, 64]
            axial_pos_embds_dim=[64, 192],  # 256
            hidden_size=256,
            chunk_size_lm_head=0,
            eos_token_id=2,
            feed_forward_size=32,
            hash_seed=None,
            hidden_act="relu",
            hidden_dropout_prob=0.05,
            initializer_range=0.02,
            is_decoder=False,
            layer_norm_eps=1e-12,
            local_num_chunks_before=1,
            local_num_chunks_after=0,
            local_attention_probs_dropout_prob=0.05,
            local_attn_chunk_length=64,
            lsh_attn_chunk_length=64,
            lsh_attention_probs_dropout_prob=0.0,
            lsh_num_chunks_before=1,
            lsh_num_chunks_after=0,
            num_attention_heads=2,
            num_buckets=4,
            num_hashes=1,
            pad_token_id=self.pad_token_id,
            vocab_size=self.vocab_size,
            tie_word_embeddings=False,
            use_cache=True,
        )
        self.generator = ReformerForMaskedLM(gen_config)
        disc_config = ReformerConfig(
            attention_head_size=64,
            attn_layers=["local", "lsh", "local", "lsh", "local", "lsh"],
            axial_norm_std=1.0,
            axial_pos_embds=True,
            axial_pos_shape=[16, 32],  # 512
            # axial_pos_shape=[64, 64], # 4096
            # max_position_embeddings=4096, # [64, 64]
            axial_pos_embds_dim=[64, 192],
            chunk_size_lm_head=0,
            eos_token_id=2,
            feed_forward_size=512,
            hash_seed=None,
            hidden_act="relu",
            hidden_dropout_prob=0.05,
            hidden_size=256,
            initializer_range=0.02,
            is_decoder=False,
            layer_norm_eps=1e-12,
            local_num_chunks_before=1,
            local_num_chunks_after=0,
            local_attention_probs_dropout_prob=0.05,
            local_attn_chunk_length=64,
            lsh_attn_chunk_length=64,
            lsh_attention_probs_dropout_prob=0.0,
            lsh_num_chunks_before=1,
            lsh_num_chunks_after=0,
            num_attention_heads=4,
            num_buckets=4,
            num_hashes=1,
            pad_token_id=self.pad_token_id,
            vocab_size=self.vocab_size,
            tie_word_embeddings=False,
            use_cache=True,
            num_labels=512,
            output_hidden_states=True,
        )
        # self.discriminator = ReformerModel(disc_config)
        self.discriminator = ReformerForSequenceClassification(disc_config)

        # tie weights of gen and disc
        self.generator.reformer.embeddings.word_embeddings = (
            self.discriminator.reformer.embeddings.word_embeddings
        )
        self.generator.reformer.embeddings.position_embeddings = (
            self.discriminator.reformer.embeddings.position_embeddings
        )
        assert (
            self.generator.reformer.embeddings.word_embeddings
            == self.discriminator.reformer.embeddings.word_embeddings
        )
        assert (
            self.generator.reformer.embeddings.position_embeddings.weights[0]
            == self.discriminator.reformer.embeddings.position_embeddings.weights[0]
        ).any()

    def forward(self, x):
        return self.discriminator(x, return_dict=True).logits

    # def mlm_step(self, gen_input):
    #     # electra generator step
    #     mlm_logits = self.generator(**gen_input, return_dict=True).logits
    #     # masked_idx = (gen_input["input_ids"] == self.mask_token_id).nonzero(
    #     #     as_tuple=True
    #     # )
    #     mlm_replacement_logits = mlm_logits[self.masked_idx]
    #     mlm_replacements = pl_data.gumbel_sample(
    #         mlm_replacement_logits, temperature=self.temperature
    #     )
    #     # generator loss
    #     self.mlm_loss = F.cross_entropy(
    #         mlm_logits.transpose(1, 2),
    #         gen_input["labels"],
    #         ignore_index=self.pad_token_id,
    #     )
    #     return mlm_replacements

    def training_step(self, batch, batch_idx):
        self.masked_idx = (batch["input_ids"] == self.mask_token_id).nonzero(
            as_tuple=True
        )

        # electra gen input
        gen_input = {k: v for k, v in batch.items()}
        gen_input["input_ids"] = batch["input_ids"].clone().detach()
        # mlm_loss = self.mlm_step(gen_input)

        # electra generator step
        mlm_logits = self.generator(**gen_input, return_dict=True).logits
        masked_idx = (gen_input["input_ids"] == self.mask_token_id).nonzero(
            as_tuple=True
        )
        mlm_replacement_logits = mlm_logits[masked_idx]
        mlm_replacements = pl_data.gumbel_sample(
            mlm_replacement_logits, temperature=self.temperature
        )

        # electra generator loss
        mlm_loss = F.cross_entropy(
            mlm_logits.transpose(1, 2),
            gen_input["labels"],
            ignore_index=self.pad_token_id,
        )

        # electra disc input
        disc_input = {k: v for k, v in batch.items() if k != "labels"}
        disc_input["input_ids"] = batch["input_ids"].clone()
        disc_input["input_ids"][self.masked_idx] = mlm_replacements.detach()
        # can't pass through classification model because it uses crossentropy
        # disc_input["labels"] = (
        #     (batch["input_ids"] != disc_input["input_ids"]).float().detach()
        # )

        # electra discriminator step
        non_padded_indices = (disc_input["input_ids"] != 0).nonzero(as_tuple=True)
        disc_logits = self.discriminator(**disc_input, return_dict=True).logits
        # disc_loss = F.binary_cross_entropy_with_logits(
        #     disc_logits[non_padded_indices], disc_input["labels"][non_padded_indices]
        # )
        disc_labels = (batch["input_ids"] != disc_input["input_ids"]).float().detach()
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[non_padded_indices], disc_labels[non_padded_indices]
        )
        loss = self.gen_weight * mlm_loss + self.disc_weight * disc_loss
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.masked_idx = (batch["input_ids"] == self.mask_token_id).nonzero(
            as_tuple=True
        )

        # electra gen input
        gen_input = {k: v for k, v in batch.items()}
        gen_input["input_ids"] = batch["input_ids"].clone().detach()
        # mlm_loss = self.mlm_step(gen_input)

        # electra generator step
        mlm_logits = self.generator(**gen_input, return_dict=True).logits
        masked_idx = (gen_input["input_ids"] == self.mask_token_id).nonzero(
            as_tuple=True
        )
        mlm_replacement_logits = mlm_logits[masked_idx]
        mlm_replacements = pl_data.gumbel_sample(
            mlm_replacement_logits, temperature=self.temperature
        )

        # electra generator loss
        mlm_loss = F.cross_entropy(
            mlm_logits.transpose(1, 2),
            gen_input["labels"],
            # ignore_index=-100,
            ignore_index=self.pad_token_id,
        )

        # electra disc input
        disc_input = {k: v for k, v in batch.items() if k != "labels"}
        disc_input["input_ids"] = batch["input_ids"].clone()
        disc_input["input_ids"][self.masked_idx] = mlm_replacements.detach()
        # can't pass through classification model because it uses crossentropy
        # disc_input["labels"] = (
        #     (batch["input_ids"] != disc_input["input_ids"]).float().detach()
        # )

        # electra discriminator step
        non_padded_indices = (disc_input["input_ids"] != 0).nonzero(as_tuple=True)
        disc_logits = self.discriminator(**disc_input, return_dict=True).logits
        # disc_loss = F.binary_cross_entropy_with_logits(
        #     disc_logits[non_padded_indices], disc_input["labels"][non_padded_indices]
        # )
        disc_labels = (batch["input_ids"] != disc_input["input_ids"]).float().detach()
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[non_padded_indices], disc_labels[non_padded_indices]
        )
        loss = self.gen_weight * mlm_loss + self.disc_weight * disc_loss
        self.log("valid_loss", loss, on_epoch=True)
        return loss

    # def configure_optimizers(self):
    #     optimizer = AdamW(
    #         self.parameters(),
    #         self.hparams.learning_rate,
    #         betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
    #         eps=self.hparams.adam_epsilon,
    #     )
    #     return optimizer

    def configure_optimizers(self):
        if not self.optimizers:
            self.optimizers = [
                torch.optim.AdamW(
                    self.parameters(),
                    lr=self.lr,
                    betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                    eps=self.hparams.adam_epsilon,
                )
            ]
        if not self.schedulers:
            self.schedulers = [
                {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizers[0], 5, eta_min=0, last_epoch=-1, verbose=True
                    ),
                    "monitor": "loss/val",  # could change to val_hit_rate
                    "interval": "epoch",
                    "frequency": 1,
                }
            ]
        return self.optimizers, self.schedulers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument("--learning_rate", type=float, default=5e-5)
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.999)
        parser.add_argument("--adam_epsilon", type=float, default=1e-8)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="distilbert-base-cased"
    )
    parser.add_argument(
        "--train_file", type=str, default="data/wikitext-2/wiki.train.small.raw"
    )
    parser.add_argument(
        "--validation_file", type=str, default="data/wikitext-2/wiki.valid.small.raw"
    )
    parser.add_argument("--line_by_line", action="store_true", default=False)
    parser.add_argument("--pad_to_max_length", action="store_true", default=False)
    parser.add_argument("--preprocessing_num_workers", type=int, default=4)
    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    parser.add_argument("--max_seq_length", type=int, default=32)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--dataloader_type", type=str, default="original")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = pl_data.McBtTrades.add_model_specific_args(parser)
    # parser = LMModel.add_model_specific_args(parser)
    parser = LMElectraModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    if args.dataloader_type == "original":
        # ['attention_mask', 'input_ids', 'labels']
        # attention_mask 1,1,1,1,1,1
        # input_ids
        data_module = LMDataModule(
            model_name_or_path=args.model_name_or_path,
            train_file=args.train_file,
            validation_file=args.validation_file,
            line_by_line=args.line_by_line,
            pad_to_max_length=args.pad_to_max_length,
            preprocessing_num_workers=args.preprocessing_num_workers,
            overwrite_cache=args.overwrite_cache,
            max_seq_length=args.max_seq_length,
            mlm_probability=args.mlm_probability,
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
        )
        # self.tokenizer.vocab["[MASK]"]
        # 103
        # self.tokenizer.vocab["[PAD]"]
        # 0

    else:
        data_module = pl_data.McBtTrades(
            mc_bt_path=args.mc_bt_path,
            bin_cols=args.bin_cols,
            symbols=args.symbols,
            model_name_or_path=args.model_name_or_path,
            pad_to_max_length=args.pad_to_max_length,
            max_seq_length=args.max_seq_length,
            mlm_probability=args.mlm_probability,
            dupe_factor=args.dupe_factor,
            short_seq_prob=args.short_seq_prob,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # ------------
    # model
    # ------------
    # lmmodel = LMModel(
    #     model_name_or_path=args.model_name_or_path,
    #     learning_rate=args.learning_rate,
    #     adam_beta1=args.adam_beta1,
    #     adam_beta2=args.adam_beta2,
    #     adam_epsilon=args.adam_epsilon,
    # )
    lmmodel = LMElectraModel(
        model_name_or_path=args.model_name_or_path,
        lr=args.lr,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(lmmodel, data_module)


if __name__ == "__main__":
    cli_main()
