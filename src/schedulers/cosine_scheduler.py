from torch import optim


class CustomScheduler:
    def __init__(
        self,
        lr: float,
        warmup_ratio: float,
        eta_min_ratio: float,
    ) -> None:
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.eta_min_ratio = eta_min_ratio
        self.warmup_steps = None

    def __call__(
        self,
        total_steps: int,
        optimizer: optim.AdamW,
    ):
        self.warmup_steps = int(total_steps * self.warmup_ratio)
        t_max = total_steps - self.warmup_steps
        eta_min = self.lr * self.eta_min_ratio

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            self.lr_lambda,
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                warmup_scheduler,
                main_scheduler,
            ],
            milestones=[
                self.warmup_steps,
            ],
        )
        return scheduler

    def lr_lambda(
        self,
        current_step: int,
    ):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return 1.0
