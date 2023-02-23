# Pecan Growth Tracking AI

## Introduction
This repository contains code for using Mask R-CNN models to track the growth of pecans. The models can be used to identify the areas of the shell, shuck, and embryo of pecans at different stages of growth. However, a manual input is required to convert the identified areas into millimeters. This can be done using a metric tape that should be placed inside each picture containing the pecans.

The code was used to conduct research on the growth of pecans, which is described in detail in the paper "Measuring pecan nut growth utilizing machine vision and deep learning for the better understanding of the fruit growth curve" by Costa et al. (2021). The results of this research were also featured in an article in Pecan South Magazine (link provided in the README).

## Magazine Article
Check out Pecan South Magazine for an article on this research. (https://www.pecansouthmagazine.com/magazine/article/know-your-nuts-from-flowering-to-fruiting/)

## Overview
The code is organized into three main parts:

1. Training: Code used to train the deep learning models for matured pecans (big) and young pecans (small).
2. pecan-mask-rcnn: Code for detecting pecan shell, shuck, and embryo based on the trained models.
3. Models: Pre-trained models for matured and young pecans. Link: https://drive.google.com/file/d/1FIemGwcc8POSeFXm7VeBQN38uQ6jZh3i/view?usp=share_link

## Research
This code was used for the following paper:

Costa, L., Ampatzidis, Y., Rohla, C., Maness, N., Cheary, B. and Zhang, L., 2021. Measuring pecan nut growth utilizing machine vision and deep learning for the better understanding of the fruit growth curve. Computers and Electronics in Agriculture, 181, p.105964.

## Contributors
The code was written by Lucas Costa, with the help of Vitor Gontijo.
