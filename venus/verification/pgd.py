# ************
# File: pgd.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Projected Gradient Descent.
# ************

import torch

from venus.split.split_strategy import SplitStrategy

class ProjectedGradientDescent:
    def __init__(self, config):
        self.config = config

    def start(self, prob, init_adv: torch.tensor=None, device=torch.device('cpu')):
        """
        PGD (see Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
        
        Arguments:
            prob:
                Verification Problem.
            init_adv:
                Initial candidate adversarial example.
        Returns:
           A tensor for the adversarial example.
        """
        # Step size to attack iterations.
        eps_iter = prob.spec.input_node.bounds.get_range() / \
            self.config.VERIFIER.PGD_EPS
        # Number of attack iterations
        num_iter = self.config.VERIFIER.PGD_NUM_ITER

        # Generate a uniformly random tensor within the specification bounds.
        if init_adv is None:
            adv = self.generate_random_adv(prob.spec.input_node.bounds)
        else:
            adv = init_adv

        i = 0
        while i < num_iter:
            adv = self.fast_gradient_signed(
                prob,
                adv,
                eps_iter,
                device
            )

            adv = torch.clamp(
                adv,
                prob.spec.input_node.bounds.lower,
                prob.spec.input_node.bounds.upper
            )

            output = prob.nn.forward(adv)
            # to edit code
            if prob.spec.is_satisfied(output, output) is not True:
                return adv.detach()

            i += 1

        return None

    def generate_random_adv(self, bounds):
        adv = torch.zeros_like(bounds.lower)
        idxs = bounds.lower < bounds.upper
        distribution = torch.distributions.uniform.Uniform(
            bounds.lower[idxs], bounds.upper[idxs]
        )
        partial_adv = distribution.sample(torch.Size([1]))
        partial_adv = torch.squeeze(partial_adv, 0)

        adv[idxs] = partial_adv

        return adv

    def fast_gradient_signed(
        self,
        prob,
        x,
        eps,
        device=torch.device('cpu'),
        specs=None
    ):
        """
        Fast Gradient Signed Method.

        Arguments: 
            prob:
                Verification Problem.
            x:
                Input tensor.
            eps:
                Epsilon.
            targeted:
                Whether or not the attack is targeted.
        Returns: 
            A tensor for the adversarial example.
        """
        x = x.clone().detach().to(self.config.PRECISION).requires_grad_(True)

        true_label = prob.spec.is_adversarial_robustness()

        save_gradient = SplitStrategy.does_node_split(
            self.config.SPLITTER.SPLIT_STRATEGY
        )
        if true_label == -1:
            output = prob.nn.forward(x, save_gradient=save_gradient).flatten()
            loss = prob.spec.get_mse_loss(output)

        else:
            output_flag =  prob.spec.get_output_flag(prob.nn.tail.output_shape)
            output = prob.nn.forward(x, save_gradient=save_gradient)[output_flag].flatten()[None, :]
            true_label = torch.sum(output_flag.flatten()[0: true_label])
            y = torch.tensor([true_label], device=device)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output, y)

        # Compute gradient
        # loss = -loss
        loss.backward()

        # compute perturbation
        perturbation = eps * torch.sign(x.grad)
        
        if torch.all(perturbation == 0):
            adv = self.generate_random_adv(prob.spec.input_node.bounds)

        else:
            adv = torch.clamp(
                x + perturbation,
                prob.spec.input_node.bounds.lower,
                prob.spec.input_node.bounds.upper
            )


        return adv
