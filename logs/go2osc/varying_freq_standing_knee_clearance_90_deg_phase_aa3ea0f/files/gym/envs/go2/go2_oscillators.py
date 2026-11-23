import torch
import math


def phase_rate(phi, grnd_force, omega, sigma, xi):
#Phase rate equation as defined in the proposal
    return 2 * math.pi * ( omega - sigma * grnd_force * (torch.cos(phi) + xi))

def step(phi, grnd_force, omega, sigma, xi, dt):
#Euler integration of the oscillator phase
    dphi = phase_rate(phi, grnd_force, omega, sigma, xi)
    return (phi + dt * dphi) % (2 * math.pi)

def desired_force(phi, amplitude):
#Desired force as defined in the proposal. 
# torch.relu removes all negative values.
# -torch.sin(phi) makes sinusoid positive during stance
# amplitude * determines the maximum force
    return amplitude * torch.relu(-torch.sin(phi))