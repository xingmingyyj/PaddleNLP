# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from functools import partial

import paddle
from datasets import load_dataset
from utils import (
    CrossEntropyLossForSQuAD,
    DataArguments,
    ModelArguments,
    QuestionAnsweringTrainer,
    load_config,
    prepare_train_features,
    prepare_validation_features,
)

import paddlenlp
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.metrics.squad import compute_prediction, squad_evaluate
from paddlenlp.trainer import (
    EvalPrediction,
    PdArgumentParser,
    TrainingArguments,
    get_last_checkpoint,
)
from paddlenlp.transformers import ErnieForQuestionAnswering, ErnieTokenizer
from paddlenlp.utils.log import logger


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load model and data config
    model_args, data_args, training_args = load_config(
        model_args.config, "QuestionAnswering", data_args.dataset, model_args, data_args, training_args
    )
    # Print model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    data_args.dataset = data_args.dataset.strip()
    training_args.output_dir = os.path.join(training_args.output_dir, data_args.dataset)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    raw_datasets = load_dataset("clue", data_args.dataset)
    label_list = getattr(raw_datasets["train"], "label_list", None)
    data_args.label_list = label_list

    # Define tokenizer, model, loss function.
    tokenizer = ErnieTokenizer.from_pretrained(model_args.model_name_or_path)
    model = ErnieForQuestionAnswering.from_pretrained(model_args.model_name_or_path)

    loss_fct = CrossEntropyLossForSQuAD()

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            # Dataset pre-process
            train_dataset = train_dataset.map(
                partial(prepare_train_features, tokenizer=tokenizer, args=data_args),
                batched=True,
                num_proc=1,
                batch_size=4,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        eval_examples = raw_datasets["validation"]
        with training_args.main_process_first(desc="evaluate dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                partial(prepare_validation_features, tokenizer=tokenizer, args=data_args),
                batched=True,
                num_proc=1,
                batch_size=4,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
    if training_args.do_predict:
        predict_examples = raw_datasets["validation"]
        contexts = predict_examples["context"]
        questions = predict_examples["question"]
        with training_args.main_process_first(desc="test dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                partial(prepare_validation_features, tokenizer=tokenizer, args=data_args),
                batched=True,
                num_proc=1,
                batch_size=4,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Define data collector
    data_collator = DataCollatorWithPadding(tokenizer)

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions, all_nbest_json, scores_diff_json = compute_prediction(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
        )

        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=predictions, label_ids=references)

    def compute_metrics(p: EvalPrediction):
        ret = squad_evaluate(examples=p.label_ids, preds=p.predictions, is_whitespace_splited=False)
        return dict(ret)
        # return metric.compute(predictions=p.predictions, references=p.label_ids)

    trainer = QuestionAnsweringTrainer(
        model=model,
        criterion=loss_fct,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if training_args.do_train:
        # Training
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate and tests model
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)

    if training_args.do_predict:
        test_ret = trainer.predict(predict_dataset, predict_examples)
        trainer.log_metrics("predict", test_ret.metrics)

        out_dict = {"answer": test_ret.predictions, "context": contexts, "question": questions}
        out_file = open(os.path.join(training_args.output_dir, "test_results.json"), "w", encoding="utf8")
        json.dump(out_dict, out_file, ensure_ascii=True)

    # Export inference model
    if training_args.do_export:
        # You can also load from certain checkpoint
        # trainer.load_state_dict_from_checkpoint("/path/to/checkpoint/")
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # segment_ids
        ]

        model_args.export_model_dir = os.path.join(model_args.export_model_dir, data_args.dataset, "export")

        paddlenlp.transformers.export_model(
            model=trainer.model, input_spec=input_spec, path=model_args.export_model_dir
        )
        trainer.tokenizer.save_pretrained(model_args.export_model_dir)


if __name__ == "__main__":
    main()
