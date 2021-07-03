import torch
from transformers import WEIGHTS_NAME, AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from tqdm import tqdm
import os


class Trainer(object):
    def __init__(self, args, model, dataloader, num_train_steps):
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.num_train_steps = num_train_steps

        self.optimizer = self._create_optimizer(model)
        self.scheduler = self._create_scheduler(self.optimizer)

    def train(self):
        model = self.model

        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model
            )

        epoch = 0
        global_step = 0
        tr_loss = 0.0

        model.train()
        with tqdm(total=self.num_train_steps, disable=self.args.local_rank not in (-1, 0)) as pbar:
            while True:
                for step, batch in enumerate(self.dataloader):
                    inputs = {k: v.to(self.args.device) for k, v in Trainer._create_model_arguments(batch).items()}
                    outputs = model(**inputs)
                    loss = outputs.loss

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    loss.backward()
                    tr_loss += loss.item()

                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        if self.args.max_grad_norm != 0.0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        self.optimizer.step()
                        self.scheduler.step()
                        model.zero_grad()

                        pbar.set_description("epoch: %d loss: %.7f" % (epoch, loss.item()))
                        pbar.update()
                        global_step += 1

                        if(self.args.local_rank in (-1, 0) and self.args.output_dir
                                and self.args.save_steps > 0 and global_step % self.args.save_steps == 0):

                            output_dir = os.path.join(self.args.output_dir, "checkpoint-{}".format(global_step))

                            if hasattr(model, "module"):
                                torch.save(model.module.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
                            else:
                                torch.save(model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

                        if global_step == self.num_train_steps:
                            break
                if global_step == self.num_train_steps:
                    break

                epoch += 1

        print("global_step = {}, average loss = {}".format(global_step, tr_loss / global_step))
        return model, global_step, tr_loss / global_step

    def _create_optimizer(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(
            optimizer_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_eps,
            betas=(self.args.adam_b1, self.args.adam_b2),
            correct_bias=self.args.adam_correct_bias,
        )

    def _create_scheduler(self, optimizer):
        warmup_steps = int(self.num_train_steps * self.args.warmup_ratio)
        if self.args.lr_schedule == "warmup_linear":
            return get_linear_schedule_with_warmup(optimizer, warmup_steps, self.num_train_steps)
        if self.args.lr_schedule == "warmup_constant":
            return get_constant_schedule_with_warmup(optimizer, warmup_steps)

        raise RuntimeError("Unsupported scheduler: " + self.args.lr_schedule)

    @staticmethod
    def _create_model_arguments(batch):
        return batch


