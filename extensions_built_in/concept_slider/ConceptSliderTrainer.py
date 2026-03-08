import gc
from collections import OrderedDict
from typing import Optional, List

import torch
from torch.utils.data import ConcatDataset, DataLoader

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.config_modules import SliderTargetConfig, SliderConfigAnchors, ReferenceDatasetConfig
from toolkit.data_loader import PairedImageDataset
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.train_tools import get_torch_dtype


class ConceptSliderTrainerConfig:
    def __init__(self, **kwargs):
        self.guidance_strength: float = kwargs.get("guidance_strength", 3.0)
        self.anchor_strength: float = kwargs.get("anchor_strength", 1.0)

        # Legacy single-pair fields (preserved for backward compatibility)
        self.positive_prompt: str = kwargs.get("positive_prompt", "")
        self.negative_prompt: str = kwargs.get("negative_prompt", "")
        self.target_class: str = kwargs.get("target_class", "")
        self.anchor_class: Optional[str] = kwargs.get("anchor_class", None)

        # --- Phase 1: Multi-target support ---
        raw_targets = kwargs.get("targets", [])
        if raw_targets:
            self.targets: List[SliderTargetConfig] = [
                SliderTargetConfig(**t) for t in raw_targets
            ]
        else:
            # Backward compat: synthesize single-item list from legacy fields
            self.targets = [
                SliderTargetConfig(
                    target_class=self.target_class,
                    positive=self.positive_prompt,
                    negative=self.negative_prompt,
                    weight=1.0,
                )
            ]

        # --- Phase 1: Multi-anchor support ---
        raw_anchors = kwargs.get("anchors", [])
        if raw_anchors:
            self.anchors: List[SliderConfigAnchors] = [
                SliderConfigAnchors(**a) for a in raw_anchors
            ]
        elif self.anchor_class is not None:
            self.anchors = [SliderConfigAnchors(prompt=self.anchor_class, multiplier=1.0)]
        else:
            self.anchors = []

        # --- Phase 2: Image pair dataset support ---
        self.datasets: List[ReferenceDatasetConfig] = [
            ReferenceDatasetConfig(**d) for d in kwargs.get("datasets", [])
        ]
        self.img_loss_weight: float = kwargs.get("img_loss_weight", 1.0)
        self.cfg_loss_weight: float = kwargs.get("cfg_loss_weight", 1.0)


def norm_like_tensor(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalize the tensor to have the same mean and std as the target tensor."""
    tensor_mean = tensor.mean()
    tensor_std = tensor.std()
    target_mean = target.mean()
    target_std = target.std()
    normalized_tensor = (tensor - tensor_mean) / (
        tensor_std + 1e-8
    ) * target_std + target_mean
    return normalized_tensor


class ConceptSliderTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.do_guided_loss = True

        self.slider: ConceptSliderTrainerConfig = ConceptSliderTrainerConfig(
            **self.config.get("slider", {})
        )

        # Phase 1: per-target embed cache
        # Each entry: { 'positive', 'negative', 'target_class', 'weight', 'guidance_strength' }
        self.prompt_pair_embeds: List[dict] = []

        # Phase 1: per-anchor embed cache
        self.anchor_embeds: List[PromptEmbeds] = []

        # Phase 2: image pair dataset state
        self.data_loader: Optional[DataLoader] = None
        self.dataset_prompts: List[str] = []
        self.dataset_prompt_cache: dict = {}
        self.train_with_dataset: bool = len(self.slider.datasets) > 0
        self._dataset_iter = None  # persistent cycling iterator

    # ------------------------------------------------------------------
    # Phase 2: dataset loading (ported from UltimateSliderTrainerProcess)
    # ------------------------------------------------------------------
    def load_datasets(self):
        if self.data_loader is not None:
            return
        if not self.slider.datasets:
            return

        print("Loading image pair datasets")
        datasets = []
        for dataset_cfg in self.slider.datasets:
            print(f" - Dataset: {dataset_cfg.pair_folder or dataset_cfg.pos_folder}")
            config = {
                "path": dataset_cfg.pair_folder,
                "size": dataset_cfg.size,
                "default_prompt": dataset_cfg.target_class,
                "network_weight": dataset_cfg.network_weight,
                "pos_weight": dataset_cfg.pos_weight,
                "neg_weight": dataset_cfg.neg_weight,
                "pos_folder": dataset_cfg.pos_folder,
                "neg_folder": dataset_cfg.neg_folder,
            }
            image_dataset = PairedImageDataset(config)
            datasets.append(image_dataset)
            self.dataset_prompts += image_dataset.get_all_prompts()

        concatenated = ConcatDataset(datasets)
        self.data_loader = DataLoader(
            concatenated,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=2,
        )

    # ------------------------------------------------------------------
    def hook_before_train_loop(self):
        # do this before calling parent as it unloads the text encoder if requested
        if self.is_caching_text_embeddings:
            # make sure model is on cpu for this part so we don't oom.
            self.sd.unet.to("cpu")

        # --- Phase 1: encode all prompt pairs and anchors ---
        with torch.no_grad():
            for target in self.slider.targets:
                pair = {
                    "positive": self.sd.encode_prompt([target.positive])
                        .to(self.device_torch, dtype=self.sd.torch_dtype)
                        .detach(),
                    "negative": self.sd.encode_prompt([target.negative])
                        .to(self.device_torch, dtype=self.sd.torch_dtype)
                        .detach(),
                    "target_class": self.sd.encode_prompt([target.target_class])
                        .to(self.device_torch, dtype=self.sd.torch_dtype)
                        .detach(),
                    "weight": target.weight,
                    "guidance_strength": self.slider.guidance_strength,
                }
                self.prompt_pair_embeds.append(pair)

            for anchor in self.slider.anchors:
                embed = (
                    self.sd.encode_prompt([anchor.prompt])
                    .to(self.device_torch, dtype=self.sd.torch_dtype)
                    .detach()
                )
                self.anchor_embeds.append(embed)

        # --- Phase 2: load image pair datasets and cache their prompts ---
        if self.train_with_dataset:
            self.load_datasets()
            with torch.no_grad():
                for prompt in self.dataset_prompts:
                    self.dataset_prompt_cache[prompt] = (
                        self.sd.encode_prompt([prompt])
                        .to(self.device_torch, dtype=self.sd.torch_dtype)
                        .detach()
                    )

        # call parent (may unload text encoder)
        super().hook_before_train_loop()

    # ------------------------------------------------------------------
    def get_guided_loss(
        self,
        noisy_latents: torch.Tensor,
        conditional_embeds: PromptEmbeds,
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: "DataLoaderBatchDTO",
        noise: torch.Tensor,
        unconditional_embeds: Optional[PromptEmbeds] = None,
        **kwargs,
    ):
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False

        dtype = get_torch_dtype(self.train_config.dtype)
        n_anchors = len(self.anchor_embeds)
        num_embeds = 3 + n_anchors  # positive, target_class, negative, anchor_0..N

        # Accumulate losses (detached) across all prompt pairs for logging only.
        # Real gradients are accumulated via .backward() inside the loop.
        total_loss_accum = torch.tensor(0.0)
        total_weight = 0.0

        # ---------------------------------------------------------------
        # Phase 1: CFG slider loss — loop over all prompt pairs
        # ---------------------------------------------------------------
        for pair in self.prompt_pair_embeds:
            pair_weight = pair["weight"]
            guidance_scale = pair["guidance_strength"]

            with torch.no_grad():
                self.sd.unet.eval()
                nl = noisy_latents.to(self.device_torch, dtype=dtype).detach()
                batch_size = nl.shape[0]

                positive_embeds = concat_prompt_embeds(
                    [pair["positive"]] * batch_size
                ).to(self.device_torch, dtype=dtype)
                target_class_embeds = concat_prompt_embeds(
                    [pair["target_class"]] * batch_size
                ).to(self.device_torch, dtype=dtype)
                negative_embeds = concat_prompt_embeds(
                    [pair["negative"]] * batch_size
                ).to(self.device_torch, dtype=dtype)

                # Build batched combo: [positive, target_class, negative, anchor_0..N]
                combo_list = [positive_embeds, target_class_embeds, negative_embeds]
                anchor_embeds_batched = []
                for ae in self.anchor_embeds:
                    batched = concat_prompt_embeds([ae] * batch_size).to(
                        self.device_torch, dtype=dtype
                    )
                    combo_list.append(batched)
                    anchor_embeds_batched.append(batched)

                combo_embeds = concat_prompt_embeds(combo_list)

                combo_pred = self.sd.predict_noise(
                    latents=torch.cat([nl] * num_embeds, dim=0),
                    conditional_embeddings=combo_embeds,
                    timestep=torch.cat([timesteps] * num_embeds, dim=0),
                    guidance_scale=1.0,
                    guidance_embedding_scale=1.0,
                    batch=batch,
                )

                # Chunk results: [positive_pred, neutral_pred, negative_pred, anchor_0..N]
                chunks = combo_pred.chunk(num_embeds, dim=0)
                positive_pred = chunks[0]
                neutral_pred = chunks[1]
                negative_pred = chunks[2]
                anchor_preds = list(chunks[3:])  # len == n_anchors

                # Compute contrastive targets
                positive = (positive_pred - neutral_pred) - (negative_pred - neutral_pred)
                negative = (negative_pred - neutral_pred) - (positive_pred - neutral_pred)

                enhance_positive_target = neutral_pred + guidance_scale * positive
                enhance_negative_target = neutral_pred + guidance_scale * negative
                erase_negative_target = neutral_pred - guidance_scale * negative
                erase_positive_target = neutral_pred - guidance_scale * positive

                # Normalize to neutral std/mean
                enhance_positive_target = norm_like_tensor(enhance_positive_target, neutral_pred)
                enhance_negative_target = norm_like_tensor(enhance_negative_target, neutral_pred)
                erase_negative_target = norm_like_tensor(erase_negative_target, neutral_pred)
                erase_positive_target = norm_like_tensor(erase_positive_target, neutral_pred)

                if was_unet_training:
                    self.sd.unet.train()

                # Restore network before the gradient-enabled passes
                if self.network is not None:
                    self.network.is_active = was_network_active

                # Build embeds for gradient-enabled passes.
                # Layout: [target_class, anchor_0, anchor_1, ...]
                if n_anchors > 0:
                    grad_embed_list = [target_class_embeds] + anchor_embeds_batched
                    grad_embeds = concat_prompt_embeds(grad_embed_list)
                    nl_grad = torch.cat([nl] * (1 + n_anchors), dim=0).to(
                        self.device_torch, dtype=dtype
                    )
                    ts_grad = torch.cat([timesteps] * (1 + n_anchors), dim=0)
                else:
                    grad_embeds = target_class_embeds.to(self.device_torch, dtype=dtype)
                    nl_grad = nl
                    ts_grad = timesteps

            # --- Positive multiplier pass ---
            self.network.set_multiplier(1.0)
            pred_pos = self.sd.predict_noise(
                latents=nl_grad,
                conditional_embeddings=grad_embeds,
                timestep=ts_grad,
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
            )

            if n_anchors > 0:
                pos_chunks = pred_pos.chunk(1 + n_anchors, dim=0)
                class_pred_pos = pos_chunks[0]
                anchor_preds_pos = list(pos_chunks[1:])
            else:
                class_pred_pos = pred_pos
                anchor_preds_pos = []

            enhance_loss = torch.nn.functional.mse_loss(class_pred_pos, enhance_positive_target)
            erase_loss = torch.nn.functional.mse_loss(class_pred_pos, erase_negative_target)

            if n_anchors > 0:
                anchor_loss_vals = [
                    torch.nn.functional.mse_loss(ap, ref)
                    for ap, ref in zip(anchor_preds_pos, anchor_preds)
                ]
                anchor_loss = sum(anchor_loss_vals) / len(anchor_loss_vals)
            else:
                anchor_loss = torch.zeros_like(erase_loss)
            anchor_loss = anchor_loss * self.slider.anchor_strength

            total_pos_loss = (enhance_loss + erase_loss + anchor_loss) / 3.0
            # Scale by pair weight and cfg_loss_weight before backward
            (total_pos_loss * pair_weight * self.slider.cfg_loss_weight).backward()
            total_pos_loss = total_pos_loss.detach()

            # --- Negative multiplier pass ---
            self.network.set_multiplier(-1.0)
            pred_neg = self.sd.predict_noise(
                latents=nl_grad,
                conditional_embeddings=grad_embeds,
                timestep=ts_grad,
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
            )

            if n_anchors > 0:
                neg_chunks = pred_neg.chunk(1 + n_anchors, dim=0)
                class_pred_neg = neg_chunks[0]
                anchor_preds_neg = list(neg_chunks[1:])
            else:
                class_pred_neg = pred_neg
                anchor_preds_neg = []

            enhance_loss = torch.nn.functional.mse_loss(class_pred_neg, enhance_negative_target)
            erase_loss = torch.nn.functional.mse_loss(class_pred_neg, erase_positive_target)

            if n_anchors > 0:
                anchor_loss_vals = [
                    torch.nn.functional.mse_loss(ap, ref)
                    for ap, ref in zip(anchor_preds_neg, anchor_preds)
                ]
                anchor_loss = sum(anchor_loss_vals) / len(anchor_loss_vals)
            else:
                anchor_loss = torch.zeros_like(erase_loss)
            anchor_loss = anchor_loss * self.slider.anchor_strength

            total_neg_loss = (enhance_loss + erase_loss + anchor_loss) / 3.0
            (total_neg_loss * pair_weight * self.slider.cfg_loss_weight).backward()
            total_neg_loss = total_neg_loss.detach()

            self.network.set_multiplier(1.0)

            pair_loss = (total_pos_loss + total_neg_loss) / 2.0
            total_loss_accum = total_loss_accum + pair_loss * pair_weight
            total_weight += pair_weight

        # Weighted average across all pairs (for logging)
        cfg_loss_log = total_loss_accum / total_weight if total_weight > 0 else total_loss_accum

        # ---------------------------------------------------------------
        # Phase 2: Image pair loss
        # ---------------------------------------------------------------
        img_loss_log = torch.tensor(0.0)

        if self.train_with_dataset and self.data_loader is not None:
            # Draw next batch from cycling iterator
            if self._dataset_iter is None:
                self._dataset_iter = iter(self.data_loader)
            try:
                img_batch = next(self._dataset_iter)
            except StopIteration:
                self._dataset_iter = iter(self.data_loader)
                img_batch = next(self._dataset_iter)

            imgs, prompts, network_weights = img_batch
            network_pos_weight, network_neg_weight = network_weights

            if isinstance(network_pos_weight, torch.Tensor):
                network_pos_weight = network_pos_weight.item()
            if isinstance(network_neg_weight, torch.Tensor):
                network_neg_weight = network_neg_weight.item()

            imgs = imgs.to(self.device_torch, dtype=dtype)
            # Split side-by-side: left=negative, right=positive
            negative_images, positive_images = torch.chunk(imgs, 2, dim=3)

            height = positive_images.shape[2]
            width = positive_images.shape[3]
            img_batch_size = positive_images.shape[0]

            with torch.no_grad():
                # Encode to latents — pass as list of individual tensors
                pos_list = [positive_images[i] for i in range(img_batch_size)]
                neg_list = [negative_images[i] for i in range(img_batch_size)]
                positive_latents = self.sd.encode_images(pos_list).to(
                    self.device_torch, dtype=dtype
                )
                negative_latents = self.sd.encode_images(neg_list).to(
                    self.device_torch, dtype=dtype
                )

                # Independent noise and timestep for image pairs
                noise_img = self.sd.get_latent_noise(
                    pixel_height=height,
                    pixel_width=width,
                    batch_size=img_batch_size,
                    noise_offset=self.train_config.noise_offset,
                ).to(self.device_torch, dtype=dtype)

                # Sample random timestep
                max_ts = self.sd.noise_scheduler.config.num_train_timesteps
                ts_img = torch.randint(
                    0, max_ts, (img_batch_size,), device=self.device_torch
                ).long()

                # Add noise (forward diffusion)
                noisy_pos = self.sd.noise_scheduler.add_noise(
                    positive_latents, noise_img, ts_img
                )
                noisy_neg = self.sd.noise_scheduler.add_noise(
                    negative_latents, noise_img, ts_img
                )

                # Concatenate so LoRA can run [+weight, -weight] in one call
                noisy_combined = torch.cat([noisy_pos, noisy_neg], dim=0)
                ts_combined = torch.cat([ts_img, ts_img], dim=0)
                noise_combined = torch.cat([noise_img, noise_img], dim=0)

                # Look up prompt embeddings from cache
                embedding_list = []
                fallback_key = next(iter(self.dataset_prompt_cache)) if self.dataset_prompt_cache else None
                for prompt in prompts:
                    emb = self.dataset_prompt_cache.get(
                        prompt,
                        self.dataset_prompt_cache.get(fallback_key) if fallback_key else None,
                    )
                    embedding_list.append(emb)
                # Double up for the [pos, neg] sides
                img_cond_embeds = concat_prompt_embeds(embedding_list + embedding_list)

            # Forward with LoRA at [+weight, -weight]
            self.network.set_multiplier([network_pos_weight, -abs(network_neg_weight)])
            noise_pred = self.sd.predict_noise(
                latents=noisy_combined,
                conditional_embeddings=img_cond_embeds,
                timestep=ts_combined,
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
            )

            if self.sd.prediction_type == "v_prediction":
                target = self.sd.noise_scheduler.get_velocity(
                    noisy_combined, noise_combined, ts_combined
                )
            else:
                target = noise_combined

            img_loss = torch.nn.functional.mse_loss(
                noise_pred.float(), target.float()
            )
            img_loss = img_loss * self.slider.img_loss_weight
            img_loss.backward()
            img_loss_log = img_loss.detach()

            self.network.set_multiplier(1.0)

            torch.cuda.empty_cache()
            gc.collect()

        # ---------------------------------------------------------------
        # Restore state and return combined scalar for logging
        # (real gradients already accumulated via .backward() calls above)
        # ---------------------------------------------------------------
        if was_unet_training:
            self.sd.unet.train()
        if self.network is not None:
            self.network.is_active = was_network_active

        total_loss = cfg_loss_log + img_loss_log

        # add a grad so backward works right (parent expects a grad-able tensor)
        total_loss.requires_grad_(True)
        return total_loss
