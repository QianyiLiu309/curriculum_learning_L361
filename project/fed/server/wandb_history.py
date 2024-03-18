"""History class which sends metrics to wandb.

Metrics are collected only at the central server, minimizing communication costs. Metric
collection only happens if wandb is turned on.
"""

from flwr.common.typing import Scalar
from flwr.server.history import History

import wandb


class WandbHistory(History):
    """History class for training and/or evaluation metrics collection."""

    def __init__(self, use_wandb: bool = True) -> None:
        """Initialize the history.

        Parameters
        ----------
        use_wandb : bool
            Whether to use wandb.
            Turn off to avoid communication overhead.

        Returns
        -------
        None
        """
        super().__init__()

        self.use_wandb = use_wandb

        self.loss_distributed_summary_defined = False
        self.loss_centralized_summary_defined = False
        self.metrics_distributed_fit_summary_defined = False
        self.metrics_distributed_summary_defined = False
        self.metrics_centralized_summary_defined = False

    def add_loss_distributed(
        self,
        server_round: int,
        loss: float,
    ) -> None:
        """Add one loss entry (from distributed evaluation) to history/wandb.

        Parameters
        ----------
        server_round : int
            The current server round.
        loss : float
            The loss to add.

        Returns
        -------
        None
        """
        super().add_loss_distributed(server_round, loss)
        if self.use_wandb:
            if not self.loss_distributed_summary_defined:
                wandb.define_metric("distributed_loss", summary="min")
                wandb.define_metric("distributed_loss", summary="max")
                self.loss_distributed_summary_defined = True

            wandb.log(
                {"distributed_loss": loss},
                step=server_round,
            )

    def add_loss_centralized(
        self,
        server_round: int,
        loss: float,
    ) -> None:
        """Add one loss entry (from centralized evaluation) to history/wandb.

        Parameters
        ----------
        server_round : int
            The current server round.
        loss : float
            The loss to add.

        Returns
        -------
        None
        """
        super().add_loss_centralized(server_round, loss)
        if self.use_wandb:
            if not self.loss_centralized_summary_defined:
                wandb.define_metric("centralised_loss", summary="min")
                wandb.define_metric("centralised_loss", summary="max")
                self.loss_centralized_summary_defined = True

            wandb.log(
                {"centralised_loss": loss},
                step=server_round,
            )

    def add_metrics_distributed_fit(
        self,
        server_round: int,
        metrics: dict[str, Scalar],
    ) -> None:
        """Add metrics entries (from distributed fit) to history/wandb.

        Parameters
        ----------
        server_round : int
            The current server round.
        metrics : Dict[str, Scalar]
            The metrics to add.

        Returns
        -------
        None
        """
        super().add_metrics_distributed_fit(
            server_round,
            metrics,
        )
        if self.use_wandb:
            if not self.metrics_distributed_fit_summary_defined:
                for key in metrics:
                    wandb.define_metric(key, summary="min")
                    wandb.define_metric(key, summary="max")
                self.metrics_distributed_fit_summary_defined = True
            for key in metrics:
                wandb.log(
                    {key: metrics[key]},
                    step=server_round,
                )

    def add_metrics_distributed(
        self,
        server_round: int,
        metrics: dict[str, Scalar],
    ) -> None:
        """Add metrics entries (from distributed evaluation) to history/wandb.

        Parameters
        ----------
        server_round : int
            The current server round.
        metrics : Dict[str, Scalar]
            The metrics to add.

        Returns
        -------
        None
        """
        super().add_metrics_distributed(
            server_round,
            metrics,
        )
        if self.use_wandb:
            if not self.metrics_distributed_summary_defined:
                for key in metrics:
                    wandb.define_metric(key, summary="min")
                    wandb.define_metric(key, summary="max")
                self.metrics_distributed_summary_defined = True
            for key in metrics:
                wandb.log(
                    {key: metrics[key]},
                    step=server_round,
                )

    def add_metrics_centralized(
        self,
        server_round: int,
        metrics: dict[str, Scalar],
    ) -> None:
        """Add metrics entries (from centralized evaluation) to history/wand.

        Parameters
        ----------
        server_round : int
            The current server round.
        metrics : Dict[str, Scalar]
            The metrics to add.

        Returns
        -------
        None
        """
        super().add_metrics_centralized(
            server_round,
            metrics,
        )
        if self.use_wandb:
            if not self.metrics_centralized_summary_defined:
                for key in metrics:
                    wandb.define_metric(key, summary="min")
                    wandb.define_metric(key, summary="max")
                self.metrics_centralized_summary_defined = True
            for key in metrics:
                wandb.log(
                    {key: metrics[key]},
                    step=server_round,
                )
