#!/bin/bash

# sudo insmod cfsmlp-inference.ko
sudo insmod cfsmlp-training.ko

sudo rmmod cfsmlp-training
# sudo rmmod cfsmlp-inference
