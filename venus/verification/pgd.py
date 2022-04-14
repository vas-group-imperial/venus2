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

class ProjectedGradientDescent:
    def __init__(self, config):
        self.config = config

    def start(self, prob):
        """
        PGD (see Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
        
        Arguments:
            prob:
                Verification Problem.
        Returns:
           A tensor for the adversarial example.
        """
        true_label = prob.spec.is_adversarial_robustness()
        if true_label == -1:
            return None


        # Step size to attack iterations.
        eps_iter = prob.spec.input_node.bounds.get_range() / self.config.VERIFIER.PGD_EPS
        # Number of attack iterations
        num_iter = self.config.VERIFIER.PGD_NUM_ITER

        # Generate a uniformly random tensor within the specification bounds.
        distribution = torch.distributions.uniform.Uniform(
            prob.spec.input_node.bounds.lower,
            prob.spec.input_node.bounds.upper
        )
        adv = distribution.sample(torch.Size([1]))
        adv = torch.squeeze(adv, 0)
    
        # untargeted output
        y = torch.tensor([true_label])

        i = 0
        while i < num_iter:
            adv = self.fast_gradient_signed(
                prob,
                adv,
                eps_iter,
                y=y,
                targeted=False
            )

            adv = torch.clamp(
                adv,
                prob.spec.input_node.bounds.lower,
                prob.spec.input_node.bounds.upper
            )

            logits = prob.nn.forward(adv)
            if prob.spec.is_satisfied(logits, logits) is not True:
                return adv.detach()

            i += 1

        return None

    def fast_gradient_signed(
        self,
        prob,
        x,
        eps,
        y=None,
        targeted=False
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
            y:
                The true output or the targeted output if targeted is set to true. 
            targeted:
                Whether or not the attack is targeted.
        Returns: 
            A tensor for the adversarial example.
        """

        true_label = prob.spec.is_adversarial_robustness()
        if true_label == -1:
            raise NotImplementedError("PGD is supported only for Linf adversarial robustness")

        assert torch.all(eps <= prob.spec.input_node.bounds.get_range())

        x = x.clone().detach().to(torch.float).requires_grad_(True)
        if y is None:
            y = torch.tensor([true_label])
       
        # Compute gradient
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(prob.nn.forward(x)[None, :], y)
        if targeted:
            loss = -loss
        loss.backward()

        # compute perturbation
        perturbation = eps * torch.sign(x.grad)

        adv = torch.clamp(
            x + perturbation,
            prob.spec.input_node.bounds.lower,
            prob.spec.input_node.bounds.upper
        )

        return adv
